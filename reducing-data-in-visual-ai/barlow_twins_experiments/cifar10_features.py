"""CIFAR-10 feature extraction shared by the in-training kNN probe and the
final sklearn linear probe.

Images are upsampled 32 -> 64 with bilinear resize and normalised with the
same mean/std as Tiny ImageNet (`shared_config.tiny_imagenet_transform`),
so the encoder sees inputs in the same distribution it was pre-trained on.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIFAR10_ROOT = os.environ.get("CIFAR10_ROOT", os.path.join(_REPO_ROOT, "cifar10_data"))

# Same mean/std as shared_config.tiny_imagenet_transform (standard ImageNet stats);
# inlined here so this file doesn't depend on shared_config.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_CIFAR10_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def get_cifar10_loaders(
    batch_size: int = 256,
    num_workers: int = 4,
    train_subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_set = CIFAR10(
        root=CIFAR10_ROOT, train=True, download=True, transform=_CIFAR10_EVAL_TRANSFORM
    )
    test_set = CIFAR10(
        root=CIFAR10_ROOT, train=False, download=True, transform=_CIFAR10_EVAL_TRANSFORM
    )

    if train_subset is not None and train_subset < len(train_set):
        # Deterministic subset for the kNN probe — first N images.
        train_set = torch.utils.data.Subset(train_set, list(range(train_subset)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


_CIFAR10_NUM_CLASSES = 10


def stratified_db_val_indices(
    labels, db_size: int, val_size: int, num_classes: int
) -> tuple[list[int], list[int]]:
    """Disjoint, class-balanced (db, val) index lists over a labelled set.

    Each split gets `db_size/num_classes` and `val_size/num_classes` examples per
    class, taken in the dataset's natural order (deterministic). Shared by the
    CIFAR-10 probe (10 classes) and the Tiny-ImageNet probe (200 classes).
    """
    if db_size % num_classes != 0 or val_size % num_classes != 0:
        raise ValueError(
            f"With stratified sampling, db_size and val_size must be divisible "
            f"by {num_classes}; got db_size={db_size}, val_size={val_size}."
        )
    per_class_db = db_size // num_classes
    per_class_val = val_size // num_classes
    labels = np.asarray(labels)
    db_idx: list[int] = []
    val_idx: list[int] = []
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) < per_class_db + per_class_val:
            raise ValueError(
                f"Class {c} has only {len(class_idx)} examples but db+val "
                f"needs {per_class_db + per_class_val}."
            )
        db_idx.extend(int(i) for i in class_idx[:per_class_db])
        val_idx.extend(
            int(i) for i in class_idx[per_class_db:per_class_db + per_class_val]
        )
    return db_idx, val_idx


def get_knn_db_val_loaders(
    db_size: int = 5000,
    val_size: int = 5000,
    stratified: bool = True,
    batch_size: int = 256,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Return (db_loader, val_loader) over disjoint subsets of CIFAR-10 train.

    Used by the optional CIFAR-10 diagnostic probe (--diagnostic_cifar_probe) in
    pretrain.py; in-domain selection uses the Tiny-ImageNet probe instead. The
    CIFAR-10 test set is not touched here — it stays clean for the final linear probe.

    When `stratified=True` (default), `db` and `val` each contain
    `db_size/10` and `val_size/10` images per class respectively, giving
    exact class balance. Requires `db_size` and `val_size` divisible by 10.
    Per-class indices are taken in the dataset's natural order (deterministic).

    When `stratified=False`, falls back to prefix slicing — `db` is
    `train[:db_size]` and `val` is `train[db_size:db_size+val_size]`.
    Provided as an opt-out for debugging / reproducing the unstratified
    behaviour; not recommended for production use because CIFAR-10's on-disk
    order is approximately (but not exactly) class-balanced.
    """
    train_set = CIFAR10(
        root=CIFAR10_ROOT, train=True, download=True, transform=_CIFAR10_EVAL_TRANSFORM
    )

    if stratified:
        db_idx, val_idx = stratified_db_val_indices(
            train_set.targets, db_size, val_size, _CIFAR10_NUM_CLASSES
        )
    else:
        if db_size + val_size > len(train_set):
            raise ValueError(
                f"db_size+val_size={db_size + val_size} exceeds CIFAR-10 train "
                f"size {len(train_set)}."
            )
        db_idx = list(range(db_size))
        val_idx = list(range(db_size, db_size + val_size))

    db_subset = torch.utils.data.Subset(train_set, db_idx)
    val_subset = torch.utils.data.Subset(train_set, val_idx)

    pin = torch.cuda.is_available()
    db_loader = DataLoader(
        db_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return db_loader, val_loader


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    for images, ys in loader:
        images = images.to(device, non_blocking=True)
        emb = model.embed(images)
        feats.append(emb.detach().cpu().numpy())
        labels.append(ys.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)
