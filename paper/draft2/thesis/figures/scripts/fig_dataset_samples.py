"""Dataset sample grids: Tiny-ImageNet (64px, pre-training) and CIFAR-10
(32px resized to 64px, downstream), one row each.

Uses the locally cached copies (HF cache for Tiny-ImageNet; torchvision
cifar10_data dir). Deterministic sample choice (fixed seed).

Usage: python fig_dataset_samples.py  (writes ../dataset_samples.pdf)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/home/yolerinskiy/Studies/TUDelft/research-project/reducing-data-in-visual-ai"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_samples.pdf")
N_SAMPLES = 8
SEED = 42


def tiny_imagenet_samples():
    from datasets import load_dataset

    ds = load_dataset("Maysee/tiny-imagenet", split="train")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds), size=N_SAMPLES, replace=False)
    imgs = []
    for i in idx:
        im = ds[int(i)]["image"]
        if im.mode != "RGB":
            im = im.convert("RGB")
        imgs.append(np.asarray(im))
    return imgs


def cifar10_samples():
    from PIL import Image
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=os.path.join(REPO, "cifar10_data"), train=True, download=False)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds), size=N_SAMPLES, replace=False)
    imgs = []
    for i in idx:
        im, _ = ds[int(i)]
        # Same 32 -> 64 bilinear resize as the probe pipeline (leo_protocol_eval).
        imgs.append(np.asarray(im.resize((64, 64), Image.BILINEAR)))
    return imgs


def main() -> None:
    ti = tiny_imagenet_samples()
    cf = cifar10_samples()

    fig, axes = plt.subplots(2, N_SAMPLES, figsize=(6.4, 1.9))
    for ax, im in zip(axes[0], ti):
        ax.imshow(im)
        ax.set_axis_off()
    for ax, im in zip(axes[1], cf):
        ax.imshow(im)
        ax.set_axis_off()

    axes[0][0].set_title(
        "Tiny-ImageNet (pre-training, $64\\times64$, no labels used)",
        fontsize=9, loc="left", x=0.0)
    axes[1][0].set_title(
        "CIFAR-10 (downstream probe, $32\\times32$ shown resized to $64\\times64$)",
        fontsize=9, loc="left", x=0.0)

    fig.tight_layout(h_pad=1.2)
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
