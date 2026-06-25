"""Two-view dataset for Barlow Twins pre-training.

Reuses the group's shared shuffled-indices master list so the subsets are
bit-identical to whatever the shared dataloader produces — only the
transforms differ (single-view + normalize for the shared loader, two SSL
views here).
"""
import json
import os

import torch
from datasets import load_dataset
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from torch.utils.data import DataLoader, Dataset


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
# Historical seed-42 file (no seed suffix); the loader falls back to it for data_seed=42.
LEGACY_INDICES_PATH = os.path.join(_DATA_DIR, "tiny_imagenet_shuffled_indices.json")


def _resolve_indices_path(data_seed: int) -> str:
    """Master shuffled-indices file for a DATA seed (which images are in each split).

    Different data seeds -> different (each nested + class-balanced) subsets, enabling
    paired (model_seed, data_seed) replicates. Seed 42 uses the historical file.
    """
    seeded = os.path.join(_DATA_DIR, f"tiny_imagenet_shuffled_indices_seed{data_seed}.json")
    if os.path.exists(seeded):
        return seeded
    if data_seed == 42 and os.path.exists(LEGACY_INDICES_PATH):
        return LEGACY_INDICES_PATH  # the historical file IS the seed-42 subset
    raise FileNotFoundError(
        f"Missing shuffled-indices file for data_seed={data_seed} (looked for {seeded}"
        + (f" and {LEGACY_INDICES_PATH}" if data_seed == 42 else "")
        + f"). Generate it with: python data/generate_splits.py --data_seed {data_seed}"
    )


def build_barlow_twins_transform(
    input_size: int = 64,
    min_scale: float = 0.25,
    cj_strength: float = 1.0,
) -> BYOLTransform:
    return BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size=input_size,
            gaussian_blur=0.0,
            min_scale=min_scale,
            cj_strength=cj_strength,
        ),
        view_2_transform=BYOLView2Transform(
            input_size=input_size,
            gaussian_blur=0.0,
            min_scale=min_scale,
            cj_strength=cj_strength,
        ),
    )


class TwoViewTinyImageNet(Dataset):
    def __init__(self, hf_dataset, indices, transform):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.hf_dataset[int(self.indices[idx])]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        view1, view2 = self.transform(image)
        return view1, view2


def get_two_view_loader(
    split: int,
    batch_size: int = 256,
    num_workers: int = 4,
    input_size: int = 64,
    min_scale: float = 0.25,
    cj_strength: float = 1.0,
    data_seed: int = 42,
) -> DataLoader:
    indices_path = _resolve_indices_path(data_seed)
    with open(indices_path, "r") as f:
        master_indices = json.load(f)

    if split > len(master_indices):
        raise ValueError(
            f"Requested split={split} exceeds master index length {len(master_indices)}"
        )
    split_indices = master_indices[:split]

    raw_dataset = load_dataset("Maysee/tiny-imagenet", split="train")
    dataset = TwoViewTinyImageNet(
        hf_dataset=raw_dataset,
        indices=split_indices,
        transform=build_barlow_twins_transform(
            input_size=input_size,
            min_scale=min_scale,
            cj_strength=cj_strength,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
