
"""
Architecture unit tests for YOLO-Count Multi-Class + QVI.

Tests:
    1. Output shapes for multiple query counts.
    2. Multi-query vs separate-query consistency.
    3. Model C zero-initialization equivalence with Model B.
    4. Nonnegative density predictions.
    5. Finite outputs.

Testing strategy:
    - Mock backbone: avoids downloading YOLO weights.
    - Mock text encoder: avoids downloading CLIP weights.
    - Real ModelA/ModelB/ModelC architecture and counting heads.

Real pretrained backbone and CLIP integration are tested separately
in the GPU Jupyter notebook.
"""

import math

import pytest
import torch
import torch.nn as nn

import yolocount_mc.models as models_module

from yolocount_mc.config import Config
from yolocount_mc.models import ModelA, ModelB, ModelC


# ---------------------------------------------------------
# TEST CONFIGURATION
# ---------------------------------------------------------

TEST_CLASSES = [
    "person",
    "car",
    "dog",
    "bicycle",
    "chair",
    "cat",
    "bus",
    "truck",
    "bird",
    "horse",
    "boat",
    "train",
    "airplane",
    "bottle",
    "bench",
    "motorcycle",
    "umbrella",
    "bird",
    "sheep",
    "cow",
]


# ---------------------------------------------------------
# LIGHTWEIGHT BACKBONE
# ---------------------------------------------------------

class TinyBackbone(nn.Module):
    """
    Lightweight test backbone.

    Returns a feature map with the same channel count and
    approximate stride expected by the counting architecture.
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=32,
        **kwargs,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_ch // 2),
            nn.SiLU(),

            nn.Conv2d(
                out_ch // 2, out_ch,
                kernel_size=3,
                stride=4,
                padding=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, image):
        return self.features(image)


# ---------------------------------------------------------
# LIGHTWEIGHT TEXT ENCODER
# ---------------------------------------------------------

class TinyTextEncoder(nn.Module):
    """
    Deterministic text encoder for architecture testing.

    It produces a fixed-dimensional embedding for each query.
    It does not replace CLIP in actual research experiments.
    """

    def __init__(self, d=64):
        super().__init__()

        self.d = d

        self.projection = nn.Linear(d, d)

    def forward(self, queries, device=None):

        batch_embeddings = []

        for image_queries in queries:

            query_embeddings = []

            for query in image_queries:

                # Deterministic character-based representation.
                seed_value = sum(
                    (index + 1) * ord(character)
                    for index, character in enumerate(query)
                )

                values = [
                    math.sin(
                        (seed_value + index * 13) * 0.017
                    )
                    for index in range(self.d)
                ]

                query_embeddings.append(values)

            batch_embeddings.append(query_embeddings)

        embeddings = torch.tensor(
            batch_embeddings,
            dtype=torch.float32,
        )

        if device is not None:
            embeddings = embeddings.to(device)

        return self.projection(embeddings)


# ---------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    """Small CPU-friendly configuration."""

    return Config(
        img_size=64,
        backbone_channels=32,
        embed_dim=64,
        qvi_channels=16,
        use_qvi=False,
        device="cpu",
        output_dir=tmp_path / "outputs",
        checkpoint_dir=tmp_path / "checkpoints",
    )


@pytest.fixture
def encoder(cfg):
    return TinyTextEncoder(
        d=cfg.embed_dim
    )


@pytest.fixture(autouse=True)
def mock_backbone(monkeypatch):
    """
    Replace pretrained YOLO with a small deterministic CNN.

    This prevents network downloads during unit testing.
    """

    monkeypatch.setattr(
        models_module,
        "YoloBackbonePAFPN",
        TinyBackbone,
    )


@pytest.fixture
def image(cfg):
    torch.manual_seed(42)

    return torch.randn(
        2,
        3,
        cfg.img_size,
        cfg.img_size,
        device=cfg.device,
    )


# ---------------------------------------------------------
# TEST 1: OUTPUT SHAPES
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "k",
    [1, 2, 5, 10, 20],
)
def test_model_b_shapes(cfg, encoder, image, k):
    """Verify Model B output shapes for different K values."""

    model = ModelB(
        encoder,
        cfg,
    ).to(cfg.device).eval()

    queries = [
        TEST_CLASSES[:k],
        TEST_CLASSES[:k],
    ]

    with torch.inference_mode():

        counts, density, cls_logits = model(
            image,
            queries,
        )

    assert counts.shape == (2, k)

    assert density.ndim == 4
    assert density.shape[:2] == (2, k)

    assert cls_logits.shape == density.shape

    assert torch.isfinite(counts).all()
    assert torch.isfinite(density).all()
    assert torch.isfinite(cls_logits).all()


# ---------------------------------------------------------
# TEST 2: SINGLE-QUERY VS MULTI-QUERY CONSISTENCY
# ---------------------------------------------------------

def test_consistency_B_eq_separate_runs(
    cfg,
    encoder,
    image,
):
    """
    Verify that processing K queries together produces
    the same counts as processing each query separately.

    This catches unintended interaction across the K dimension.
    """

    model = ModelB(
        encoder,
        cfg,
    ).to(cfg.device).eval()

    single_image = image[:1]

    queries = TEST_CLASSES[:3]

    with torch.inference_mode():

        counts_multi, _, _ = model(
            single_image,
            [queries],
        )

        counts_separate = torch.stack(
            [
                model(
                    single_image,
                    [[query]],
                )[0][:, 0]

                for query in queries
            ],
            dim=1,
        )

    max_difference = (
        counts_multi - counts_separate
    ).abs().max().item()

    assert torch.allclose(
        counts_multi,
        counts_separate,
        atol=1e-4,
        rtol=1e-4,
    ), (
        "Multi-query consistency failed. "
        f"Maximum difference: {max_difference:.6e}"
    )


# ---------------------------------------------------------
# TEST 3: QVI ZERO-INITIALIZATION
# ---------------------------------------------------------

def test_qvi_zero_init_equals_B(
    cfg,
    image,
):
    """
    A freshly initialized Model C with zero-initialized
    QVI correction layers should reproduce Model B.

    Non-QVI weights are copied from B into C.
    """

    cfg_B = Config(
        **{
            **cfg.__dict__,
            "use_qvi": False,
        }
    )

    cfg_C = Config(
        **{
            **cfg.__dict__,
            "use_qvi": True,
            "qvi_type": "modulation",
        }
    )

    encoder_B = TinyTextEncoder(
        d=cfg.embed_dim
    )

    encoder_C = TinyTextEncoder(
        d=cfg.embed_dim
    )

    model_B = ModelB(
        encoder_B,
        cfg_B,
    ).to(cfg.device).eval()

    model_C = ModelC(
        encoder_C,
        cfg_C,
    ).to(cfg.device).eval()

    # Copy matching non-QVI parameters.
    state_B = model_B.state_dict()
    state_C = model_C.state_dict()

    common_weights = {
        key: value
        for key, value in state_B.items()
        if key in state_C
        and not key.startswith("qvi.")
    }

    model_C.load_state_dict(
        common_weights,
        strict=False,
    )

    queries = [
        TEST_CLASSES[:2]
    ]

    with torch.inference_mode():

        counts_B, density_B, cls_B = model_B(
            image[:1],
            queries,
        )

        counts_C, density_C, cls_C = model_C(
            image[:1],
            queries,
        )

    assert torch.allclose(
        counts_B,
        counts_C,
        atol=1e-5,
        rtol=1e-4,
    ), "QVI initialization changed the Model B counts."

    assert torch.allclose(
        density_B,
        density_C,
        atol=1e-5,
        rtol=1e-4,
    ), "QVI initialization changed the density maps."

    assert torch.allclose(
        cls_B,
        cls_C,
        atol=1e-5,
        rtol=1e-4,
    ), "QVI initialization changed the classification maps."


# ---------------------------------------------------------
# TEST 4: NONNEGATIVE DENSITY
# ---------------------------------------------------------

def test_density_predictions_nonnegative(
    cfg,
    encoder,
    image,
):
    """Density predictions must never be negative."""

    model = ModelB(
        encoder,
        cfg,
    ).to(cfg.device).eval()

    with torch.inference_mode():

        _, density, _ = model(
            image,
            [
                TEST_CLASSES[:3],
                TEST_CLASSES[:3],
            ],
        )

    assert torch.isfinite(density).all()

    assert torch.all(
        density >= 0
    ), "Negative density values detected."


# ---------------------------------------------------------
# TEST 5: MODEL A SINGLE-QUERY CONTRACT
# ---------------------------------------------------------

def test_model_A_single_query(
    cfg,
    encoder,
    image,
):
    """Model A must accept exactly one query per image."""

    model = ModelA(
        encoder,
        cfg,
    ).to(cfg.device).eval()

    queries = [
        ["person"],
        ["car"],
    ]

    with torch.inference_mode():

        counts, density, cls_logits = model(
            image,
            queries,
        )

    assert counts.shape == (2, 1)
    assert density.shape[:2] == (2, 1)
    assert cls_logits.shape[:2] == (2, 1)


# ---------------------------------------------------------
# TEST 6: INVALID QUERY INPUT
# ---------------------------------------------------------

def test_invalid_empty_query(
    cfg,
    encoder,
    image,
):
    """Empty query lists should be rejected."""

    model = ModelB(
        encoder,
        cfg,
    ).to(cfg.device).eval()

    with pytest.raises(
        (ValueError, RuntimeError)
    ):
        model(
            image,
            [[] for _ in range(image.shape[0])],
        )
