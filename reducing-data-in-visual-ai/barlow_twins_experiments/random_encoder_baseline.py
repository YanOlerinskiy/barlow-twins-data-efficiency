"""Linear-probe a freshly-initialised BarlowTwinsViT (no SSL training) on
CIFAR-10. Establishes the "split=0" floor for the learning curve.

Run from the repo root:
    python barlow_twins_experiments/random_encoder_baseline.py
"""
import os
import sys
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO_ROOT)
from models.barlow_twins import BarlowTwinsViT  # noqa: E402

from barlow_twins_experiments.cifar10_features import (  # noqa: E402
    extract_features,
    get_cifar10_loaders,
)


def main() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BarlowTwinsViT().to(device)
    model.eval()

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=256, num_workers=2, train_subset=None
    )
    print("Extracting features from random encoder...")
    train_X, train_y = extract_features(model, train_loader, device)
    test_X, test_y = extract_features(model, test_loader, device)

    print(f"Train features: {train_X.shape}, Test features: {test_X.shape}")
    print("Fitting sklearn LogisticRegression...")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(train_X, train_y)
    accuracy = float(clf.score(test_X, test_y))
    print(f"\nRandom-encoder linear-probe accuracy on CIFAR-10: {accuracy:.4f}")
    print("(This is the 'split=0' floor for the learning curve.)")


if __name__ == "__main__":
    main()
