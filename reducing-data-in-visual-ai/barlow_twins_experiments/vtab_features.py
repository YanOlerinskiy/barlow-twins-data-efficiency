"""VTAB-1k in-training kNN transfer probe.

Provides loaders for an in-training kNN over the VTAB-1k bundle: gallery = each
task's `train800` split, query = its `val200` split. The VTAB **test** split is
never touched here (it stays clean for the final linear-probe eval). Averaging the
per-task kNN accuracy gives a single transfer signal logged across pretraining,
used to show that the in-domain (Tiny-ImageNet) checkpoint selection peak differs
from the transfer peak ("selection matters").

The dataset class is the same disk-backed reader used by the linear-probe notebook
(`yan-vtab-finetuning.ipynb`), moved here so it is importable by DataLoader workers
and shared between the probe and the notebook.
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VTAB_DATA_ROOT = os.path.join(_REPO_ROOT, "vtab-1k")

# The 19 VTAB-1k tasks grouped by category (the standard VTAB split: 7 natural / 4 specialized /
# 8 structured). Bundle dir name == task label. Consumed by the analysis tooling for the
# per-group transfer breakdown.
VTAB_TASK_GROUPS = {
    "natural": [
        "caltech101", "cifar", "dtd", "oxford_flowers102", "oxford_iiit_pet", "sun397", "svhn",
    ],
    "specialized": ["eurosat", "patch_camelyon", "resisc45", "diabetic_retinopathy"],
    "structured": [
        "clevr_count", "clevr_dist", "dmlab", "kitti",
        "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele",
    ],
}
# Flat list (preserves the natural -> specialized -> structured order).
VTAB_TASKS = [t for group in VTAB_TASK_GROUPS.values() for t in group]
TASK_TO_GROUP = {t: g for g, ts in VTAB_TASK_GROUPS.items() for t in ts}

# Same mean/std + 64x64 resize as pretraining inputs and the linear-probe notebook,
# so the encoder sees the distribution it was trained on. (TI is native 64px; VTAB
# images vary in size and must be resized.)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
VTAB_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Expected train-side split sizes (cheap integrity check on load).
_SPLIT_SIZES = {"train800": 800, "val200": 200, "train800val200": 1000}


class VtabImageDataset(Dataset):
    """Disk-backed VTAB-1k split. Stores only the (image path, label) list and loads
    + transforms each image lazily via torchvision's default_loader, driven by the
    bundle's `<split>.txt` label file (lines: `images/<split>/NNNNNN.jpg <label>`)."""

    def __init__(self, root, task_dir, split, transform):
        base = os.path.join(root, task_dir)
        self.transform = transform
        with open(os.path.join(base, f"{split}.txt")) as f:
            lines = f.read().splitlines()
        self.samples = [(os.path.join(base, rel), int(lab))
                        for rel, lab in (ln.split() for ln in lines)]
        if split in _SPLIT_SIZES:
            assert len(self.samples) == _SPLIT_SIZES[split], \
                f"{task_dir}/{split}: expected {_SPLIT_SIZES[split]} images, got {len(self.samples)}"

    @property
    def num_classes(self):
        return max(lab for _, lab in self.samples) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(default_loader(path)), label


def get_vtab_knn_loaders(
    task: str,
    data_root: str = VTAB_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, int]:
    """Return (gallery_loader=train800, query_loader=val200, num_classes) for a task.

    Test split untouched. num_classes is derived from the labels (max over both splits).
    """
    gallery = VtabImageDataset(data_root, task, "train800", VTAB_EVAL_TRANSFORM)
    query = VtabImageDataset(data_root, task, "val200", VTAB_EVAL_TRANSFORM)
    num_classes = max(gallery.num_classes, query.num_classes)
    pin = torch.cuda.is_available()
    db_loader = DataLoader(gallery, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin)
    q_loader = DataLoader(query, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    return db_loader, q_loader, num_classes


def resolve_vtab_tasks(spec: str | None) -> list[str]:
    """Parse a comma-separated task spec (or None/'all') into a validated task list."""
    if spec is None or spec.strip().lower() == "all":
        return list(VTAB_TASKS)
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in VTAB_TASKS]
    if unknown:
        raise ValueError(f"Unknown VTAB task(s) {unknown}; valid: {VTAB_TASKS}")
    return tasks
