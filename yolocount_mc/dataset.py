
"""
Real multi-class counting datasets for YOLO-Count.

Supported:
    1. COCO instance annotations
    2. LVIS instance annotations

FSC147 requires a separate point/exemplar-based adapter.

Sample contract:
    image:   FloatTensor [3,H,W], values in [0,1]
    queries: List[str]
    counts:  FloatTensor [K]

Collate:
    images:  [B,3,H,W]
    queries: List[List[str]]
    counts:  [B,K]
    mask:    [B,K]
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


VOCAB: List[str] = [
    "car", "person", "dog", "chair", "cat", "bicycle"
]


class InstanceCountingDataset(Dataset):
    """
    COCO/LVIS multi-class instance-counting dataset.

    Only categories with reliable image-level labels are sampled.

    Crowd annotations are excluded from instance counts. Categories
    represented only by crowd annotations are excluded from queries
    rather than incorrectly labelled as zero.
    """

    def __init__(
        self,
        root: str,
        annotation_file: Optional[str] = None,
        image_dir: Optional[str] = None,
        dataset_name: str = "coco",
        img_size: int = 640,
        max_queries: int = 8,
        training: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.root = Path(root)
        self.dataset_name = dataset_name.lower()
        self.img_size = int(img_size)
        self.max_queries = int(max_queries)
        self.training = training
        self.seed = int(seed)

        if self.dataset_name not in {"coco", "lvis"}:
            raise ValueError(
                "Only COCO and LVIS are supported. "
                "FSC147 needs a separate adapter."
            )

        if annotation_file is None:
            raise ValueError("annotation_file must be provided.")

        if self.img_size <= 0:
            raise ValueError("img_size must be positive.")

        if self.max_queries <= 0:
            raise ValueError("max_queries must be positive.")

        self.annotation_file = Path(annotation_file)

        if not self.annotation_file.is_absolute():
            self.annotation_file = self.root / self.annotation_file

        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )

        if image_dir is None:
            self.image_dir = self.root
        else:
            self.image_dir = Path(image_dir)

            if not self.image_dir.is_absolute():
                self.image_dir = self.root / self.image_dir

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        with open(self.annotation_file, "r", encoding="utf-8") as file:
            self.data = json.load(file)

        for key in ("images", "annotations", "categories"):
            if key not in self.data:
                raise ValueError(
                    f"Annotation JSON is missing required field: {key}"
                )

        self.categories = {
            int(category["id"]): str(category["name"])
            for category in self.data["categories"]
        }

        if not self.categories:
            raise ValueError("No categories found.")

        self.images = {
            int(image["id"]): image
            for image in self.data["images"]
        }

        if not self.images:
            raise ValueError("No images found in annotation JSON.")

        self.annotations_by_image = {
            image_id: [] for image_id in self.images
        }

        self.crowd_categories_by_image = {
            image_id: set() for image_id in self.images
        }

        # -----------------------------------------------------
        # ANNOTATION INDEX
        # -----------------------------------------------------

        for annotation in self.data["annotations"]:

            try:
                image_id = int(annotation["image_id"])
                category_id = int(annotation["category_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid annotation record: {annotation}"
                ) from exc

            if image_id not in self.images:
                continue

            if category_id not in self.categories:
                continue

            if annotation.get("iscrowd", 0):
                self.crowd_categories_by_image[image_id].add(
                    category_id
                )
                continue

            self.annotations_by_image[image_id].append(annotation)

        # -----------------------------------------------------
        # VALID IMAGE INDEX
        # -----------------------------------------------------

        self.image_ids = []

        for image_id in sorted(self.images):

            positive, valid = self._get_valid_categories(image_id)

            if valid:
                self.image_ids.append(image_id)

        if not self.image_ids:
            raise ValueError(
                "No usable images found. Check annotations and paths."
            )

        print(
            f"[{self.dataset_name.upper()}] Loaded "
            f"{len(self.image_ids)} images and "
            f"{len(self.categories)} categories."
        )

    def __len__(self):
        return len(self.image_ids)

    def _get_valid_categories(self, image_id):
        """
        Return positive and reliably labelled categories.

        COCO:
            Categories are treated as exhaustively annotated,
            except categories represented only by crowd instances.

        LVIS:
            Positive categories come from non-crowd annotations.
            Explicit negative categories are valid negatives.
            Not-exhaustive categories are excluded unless positive.
            Unknown categories are never treated as zero.
        """

        annotations = self.annotations_by_image[image_id]

        positive_categories = {
            int(annotation["category_id"])
            for annotation in annotations
        }

        crowd_categories = self.crowd_categories_by_image[image_id]

        if self.dataset_name == "coco":

            valid_categories = (
                set(self.categories)
                - crowd_categories
            ) | positive_categories

        else:

            image_info = self.images[image_id]

            negative_categories = {
                int(category_id)
                for category_id in image_info.get(
                    "neg_category_ids", []
                )
            }

            not_exhaustive = {
                int(category_id)
                for category_id in image_info.get(
                    "not_exhaustive_category_ids", []
                )
            }

            # Explicitly negative and not-exhaustive cannot both
            # be used as reliable negative labels.
            valid_negative = (
                negative_categories
                - not_exhaustive
                - crowd_categories
            )

            valid_categories = (
                positive_categories | valid_negative
            )

        valid_categories = {
            category_id
            for category_id in valid_categories
            if category_id in self.categories
        }

        return positive_categories, valid_categories

    def _sample_queries(self, image_id, index):
        """
        Select positive and valid negative queries.

        Sampling is deterministic for each image index.
        Category IDs are sorted before sampling for reproducibility.
        """

        positive, valid = self._get_valid_categories(image_id)

        rng = random.Random(self.seed + index)

        positive_list = sorted(positive)
        negative_list = sorted(valid - positive)

        if len(positive_list) > self.max_queries:
            selected = rng.sample(
                positive_list,
                self.max_queries,
            )
        else:
            selected = list(positive_list)

            remaining = self.max_queries - len(selected)

            if remaining > 0 and negative_list:
                selected.extend(
                    rng.sample(
                        negative_list,
                        min(remaining, len(negative_list)),
                    )
                )

        if self.training:
            rng.shuffle(selected)

        if not selected:
            raise RuntimeError(
                f"No valid queries for image {image_id}."
            )

        return selected

    def _load_image(self, image_info):
        """
        Load RGB image and resize to a square.

        NOTE:
            Direct square resizing changes aspect ratio.
            Use this exact preprocessing policy during inference too.
        """

        image_path = self.image_dir / image_info["file_name"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image = image.resize(
                (self.img_size, self.img_size),
                Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        return image_tensor

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        image_info = self.images[image_id]

        image = self._load_image(image_info)

        selected_categories = self._sample_queries(
            image_id,
            index,
        )

        counts_by_category = {}

        for annotation in self.annotations_by_image[image_id]:

            category_id = int(annotation["category_id"])

            counts_by_category[category_id] = (
                counts_by_category.get(category_id, 0) + 1
            )

        queries = [
            self.categories[category_id]
            for category_id in selected_categories
        ]

        counts = torch.tensor(
            [
                float(counts_by_category.get(category_id, 0))
                for category_id in selected_categories
            ],
            dtype=torch.float32,
        )

        return image, queries, counts


class COCODataset(InstanceCountingDataset):
    """COCO-specific convenience wrapper."""

    def __init__(
        self,
        root,
        annotation_file,
        image_dir,
        **kwargs,
    ):
        super().__init__(
            root=root,
            annotation_file=annotation_file,
            image_dir=image_dir,
            dataset_name="coco",
            **kwargs,
        )


class LVISDataset(InstanceCountingDataset):
    """LVIS-specific convenience wrapper."""

    def __init__(
        self,
        root,
        annotation_file,
        image_dir,
        **kwargs,
    ):
        super().__init__(
            root=root,
            annotation_file=annotation_file,
            image_dir=image_dir,
            dataset_name="lvis",
            **kwargs,
        )


class FSC147Dataset(Dataset):
    """
    FSC147 requires a dedicated point/exemplar-based adapter.

    This class intentionally refuses to fabricate instance-count
    targets from an incompatible annotation format.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FSC147 uses point annotations and exemplar boxes. "
            "A dedicated adapter requires verification of the official "
            "annotation files and directory layout."
        )


def collate(batch):
    """
    Collate variable numbers of class queries.

    Returns:
        images:  [B,3,H,W]
        queries: List[List[str]]
        counts:  [B,Kmax]
        mask:    [B,Kmax]

    Padded queries duplicate a valid query.
    Their targets are zero and their mask values are zero.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    images_list = [item[0] for item in batch]
    queries_list = [item[1] for item in batch]
    counts_list = [item[2] for item in batch]

    if any(not isinstance(image, torch.Tensor) for image in images_list):
        raise TypeError("Every image must be a torch.Tensor.")

    if any(image.ndim != 3 or image.shape[0] != 3 for image in images_list):
        raise ValueError("Images must have shape [3,H,W].")

    if any(image.shape != images_list[0].shape for image in images_list):
        raise ValueError(
            "All images in a batch must have the same shape."
        )

    if any(not query for query in queries_list):
        raise ValueError("Every sample must have at least one query.")

    for query, count in zip(queries_list, counts_list):

        if len(query) != count.numel():
            raise ValueError(
                "Query count and target count dimensions do not match."
            )

        if not all(
            isinstance(name, str) and name.strip()
            for name in query
        ):
            raise ValueError("Queries must be non-empty strings.")

    images = torch.stack(images_list)

    max_queries = max(len(query) for query in queries_list)

    padded_queries = []
    padded_counts = []
    masks = []

    for query, count in zip(queries_list, counts_list):

        actual_k = len(query)
        padding = max_queries - actual_k

        padded_query = query + [query[0]] * padding

        padded_count = F.pad(
            count,
            (0, padding),
            value=0.0,
        )

        mask = torch.tensor(
            [1.0] * actual_k + [0.0] * padding,
            dtype=torch.float32,
        )

        padded_queries.append(padded_query)
        padded_counts.append(padded_count)
        masks.append(mask)

    counts = torch.stack(padded_counts)
    mask = torch.stack(masks)

    return images, padded_queries, counts, mask
