
"""
Real multi-class counting datasets for YOLO-Count.

Supported:
    1. COCO instance annotations
    2. LVIS instance annotations

FSC147 requires a separate point/exemplar-based counting adapter.
It must not be treated as a COCO/LVIS instance-annotation dataset.

Sample contract:
    image:   FloatTensor [3, H, W], values in [0, 1]
    queries: List[str], length K
    counts:  FloatTensor [K]
    collate:
        images: [B, 3, H, W]
        queries: List[List[str]]
        counts: [B, K]
        mask:   [B, K]
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


VOCAB: List[str] = [
    "car", "person", "dog", "chair", "cat", "bicycle"
]


class InstanceCountingDataset(Dataset):
    """
    COCO/LVIS multi-class counting dataset.

    For each image:
        - Count annotated instances per category.
        - Select positive categories.
        - Select valid negative categories.
        - Never label unknown LVIS categories as zero.

    Args:
        root:
            Dataset root directory.

        annotation_file:
            COCO/LVIS annotation JSON path.

        image_dir:
            Directory containing the image files.

        dataset_name:
            "coco" or "lvis".

        img_size:
            Square image size.

        max_queries:
            Maximum number of category queries per image.

        training:
            Enables random query sampling.

        seed:
            Random seed.
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
        self.img_size = img_size
        self.max_queries = max_queries
        self.training = training
        self.seed = seed

        if self.dataset_name not in {"coco", "lvis"}:
            raise ValueError(
                "InstanceCountingDataset supports only COCO and LVIS. "
                "Use a separate FSC147 adapter for point annotations."
            )

        if annotation_file is None:
            raise ValueError(
                "Please provide the actual annotation_file path."
            )

        self.annotation_file = Path(annotation_file)

        if not self.annotation_file.is_absolute():
            self.annotation_file = self.root / self.annotation_file

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )

        if image_dir is None:
            self.image_dir = self.root
        else:
            self.image_dir = Path(image_dir)

            if not self.image_dir.is_absolute():
                self.image_dir = self.root / self.image_dir

        # -----------------------------------------------------
        # LOAD ANNOTATIONS
        # -----------------------------------------------------
        with open(self.annotation_file, "r", encoding="utf-8") as file:
            self.data = json.load(file)

        self.categories = {
            int(category["id"]): category["name"]
            for category in self.data["categories"]
        }

        if not self.categories:
            raise ValueError("No categories found in annotation JSON.")

        self.images = {
            int(image["id"]): image
            for image in self.data["images"]
        }

        self.annotations_by_image = {
            image_id: []
            for image_id in self.images
        }

        for annotation in self.data["annotations"]:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])

            if image_id not in self.annotations_by_image:
                continue

            if category_id not in self.categories:
                continue

            # Ignore crowd annotations for instance counting.
            if annotation.get("iscrowd", 0):
                continue

            self.annotations_by_image[image_id].append(annotation)

        # -----------------------------------------------------
        # VALID IMAGE INDEX
        # -----------------------------------------------------
        self.image_ids = []

        for image_id, image_info in self.images.items():

            annotations = self.annotations_by_image[image_id]

            positive_categories = {
                int(annotation["category_id"])
                for annotation in annotations
            }

            if self.dataset_name == "coco":
                # COCO instance annotations are treated as exhaustive
                # for the categories in this annotation file.
                valid_categories = set(self.categories)

            else:
                # LVIS distinguishes unknown categories from known
                # negative categories.
                negative_categories = set(
                    image_info.get("neg_category_ids", [])
                )

                not_exhaustive = set(
                    image_info.get("not_exhaustive_category_ids", [])
                )

                valid_categories = (
                    positive_categories | negative_categories
                )

                valid_categories -= not_exhaustive - positive_categories

            if valid_categories:
                self.image_ids.append(image_id)

        if not self.image_ids:
            raise ValueError(
                "No usable images found. Check annotation paths "
                "and dataset split."
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
        Return categories with reliable image-level labels.

        For COCO:
            All categories are considered exhaustively annotated.

        For LVIS:
            Only positive and explicitly negative categories are used.
            Unknown categories are excluded.
        """

        annotations = self.annotations_by_image[image_id]

        positive_categories = {
            int(annotation["category_id"])
            for annotation in annotations
        }

        if self.dataset_name == "coco":
            valid_categories = set(self.categories)

        else:
            image_info = self.images[image_id]

            negative_categories = set(
                image_info.get("neg_category_ids", [])
            )

            not_exhaustive = set(
                image_info.get("not_exhaustive_category_ids", [])
            )

            valid_categories = (
                positive_categories | negative_categories
            )

            valid_categories -= not_exhaustive - positive_categories

        return positive_categories, valid_categories

    def _sample_queries(self, image_id, index):
        """
        Select positive and valid negative class queries.

        Training uses deterministic per-index random sampling.
        """

        positive, valid = self._get_valid_categories(image_id)

        rng = random.Random(self.seed + index)

        negative = valid - positive

        selected = list(positive)

        # Keep as many positive categories as possible.
        if len(selected) > self.max_queries:
            selected = rng.sample(selected, self.max_queries)

        remaining = self.max_queries - len(selected)

        if remaining > 0 and negative:
            sampled_negative = rng.sample(
                list(negative),
                min(remaining, len(negative)),
            )
            selected.extend(sampled_negative)

        # Shuffle query order during training.
        if self.training:
            rng.shuffle(selected)

        if not selected:
            raise RuntimeError(
                f"No valid category queries for image {image_id}."
            )

        return selected

    def _load_image(self, image_info):
        """
        Load RGB image and resize it to a square.

        This version uses direct square resizing. The same resizing
        policy must be used consistently during training and inference.
        """

        image_path = self.image_dir / image_info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image = image.resize(
            (self.img_size, self.img_size),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(image, dtype=np.float32) / 255.0

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

        # Count instances per selected category.
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

    def __init__(self, root, annotation_file, image_dir, **kwargs):
        super().__init__(
            root=root,
            annotation_file=annotation_file,
            image_dir=image_dir,
            dataset_name="coco",
            **kwargs,
        )


class LVISDataset(InstanceCountingDataset):
    """LVIS-specific convenience wrapper."""

    def __init__(self, root, annotation_file, image_dir, **kwargs):
        super().__init__(
            root=root,
            annotation_file=annotation_file,
            image_dir=image_dir,
            dataset_name="lvis",
            **kwargs,
        )


class FSC147Dataset(Dataset):
    """
    FSC147 requires a separate point/exemplar counting implementation.

    Its annotation structure and supervision are not interchangeable
    with COCO/LVIS instance annotations. This class intentionally
    refuses to fabricate count targets.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FSC147 uses point annotations and exemplar boxes. "
            "We will implement its dedicated adapter after verifying "
            "your official FSC147 annotation files and directory layout."
        )


def collate(batch):
    """
    Collate variable numbers of queries.

    Returns:
        images: [B, 3, H, W]
        queries: List[List[str]]
        counts: [B, Kmax]
        mask: [B, Kmax]

    Padded query strings duplicate the first real query.
    Their count targets are zero and their masks are zero.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    images = torch.stack([item[0] for item in batch])

    queries = [item[1] for item in batch]
    counts_list = [item[2] for item in batch]

    if any(len(query) == 0 for query in queries):
        raise ValueError("Every sample must contain at least one query.")

    max_queries = max(len(query) for query in queries)

    padded_queries = []
    padded_counts = []
    masks = []

    for query, count in zip(queries, counts_list):

        actual_k = len(query)
        padding = max_queries - actual_k

        # Duplicate a valid query instead of using an empty string.
        padded_query = query + [query[0]] * padding

        padded_count = torch.nn.functional.pad(
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
