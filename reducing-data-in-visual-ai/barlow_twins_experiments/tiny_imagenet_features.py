"""In-domain Tiny-ImageNet feature extraction for the in-training kNN probe.

The gallery and query sets are stratified-disjoint partitions of the labelled
Tiny-ImageNet `valid` split (10k images, 50/class, 200 classes), which is held
out from `train` for every split (including 100k) — so the probe never leaks
training images and the signal is identical across split sizes.

Tiny-ImageNet is natively 64x64, so (unlike the CIFAR probe) no resize is
needed; we reuse `shared_config.tiny_imagenet_transform` (ToTensor + ImageNet
normalisation), the same single-view pipeline the encoder saw the train images
under, and `cifar10_features.extract_features` for the forward pass.
"""
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO_ROOT)
from shared_config import tiny_imagenet_transform  # noqa: E402

from barlow_twins_experiments.cifar10_features import (  # noqa: E402
    stratified_db_val_indices,
)

_TINY_IMAGENET_NUM_CLASSES = 200


class TinyImageNetEval(Dataset):
    """Single-view (non-augmented) Tiny-ImageNet for kNN feature extraction."""

    def __init__(self, hf_dataset, indices, transform):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.hf_dataset[int(self.indices[idx])]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(item["label"])


def get_tiny_imagenet_knn_loaders(
    db_size: int = 5000,
    val_size: int = 5000,
    batch_size: int = 256,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Return (db_loader, val_loader) over disjoint, class-balanced partitions of
    the Tiny-ImageNet `valid` split.

    `db_size`/`val_size` must each be divisible by 200, and their sum must not
    exceed 10000 (the val split has exactly 50 images/class). Defaults
    (5000/5000) use all 50 images/class as 25 gallery + 25 query.
    """
    val_set = load_dataset("Maysee/tiny-imagenet", split="valid")
    labels = val_set["label"]
    db_idx, query_idx = stratified_db_val_indices(
        labels, db_size, val_size, _TINY_IMAGENET_NUM_CLASSES
    )

    db_ds = TinyImageNetEval(val_set, db_idx, tiny_imagenet_transform)
    query_ds = TinyImageNetEval(val_set, query_idx, tiny_imagenet_transform)

    pin = torch.cuda.is_available()
    db_loader = DataLoader(
        db_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        query_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return db_loader, val_loader
