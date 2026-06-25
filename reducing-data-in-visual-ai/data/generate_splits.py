import argparse
import os
import json
import numpy as np
from datasets import load_dataset


def indices_filename(seed: int, script_dir: str | None = None) -> str:
    """Path of the master shuffled-indices file for a given DATA seed.

    Different data seeds give different (but each internally nested + class-balanced)
    image subsets, so a paired (model_seed, data_seed) run varies which images are
    pretrained on, not just the training RNG. Seed 42 keeps the historical filename
    for backward compatibility; other seeds get a `_seed{N}` suffix.
    """
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    if seed == 42:
        # Historical default file (the original campaign's subset). The loader also
        # falls back to this when the _seed42 variant is absent.
        return os.path.join(script_dir, "tiny_imagenet_shuffled_indices.json")
    return os.path.join(script_dir, f"tiny_imagenet_shuffled_indices_seed{seed}.json")


def generate_tiny_imagenet_stratified_splits(seed: int = 42) -> str:
    print(f"Generating deterministic stratified splits of Tiny ImageNet (Seed: {seed})...")
    np.random.seed(seed)

    print("Loading Hugging Face dataset to map classes...")
    dataset = load_dataset("Maysee/tiny-imagenet")
    train_labels = np.array(dataset['train']['label'])

    num_classes = 200
    class_indices = {c: np.where(train_labels == c)[0] for c in range(num_classes)}

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    master_indices = []
    max_images_per_class = max(len(np.where(train_labels == c)[0]) for c in range(num_classes))

    # Interleave one image from each class iteratively
    # This ensures that any prefix of size N (where N % num_classes == 0) is perfectly stratified
    for i in range(max_images_per_class):
        class_order = np.random.permutation(num_classes)
        for c in class_order:
            if i < len(class_indices[c]):
                master_indices.append(int(class_indices[c][i]))

    filename = indices_filename(seed)
    with open(filename, "w") as f:
        json.dump(master_indices, f)

    print(f"Saved {len(master_indices)} master stratified indices to {filename}")
    return filename


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a stratified, nested Tiny-ImageNet "
                                             "index file for a given DATA seed.")
    ap.add_argument("--data_seed", type=int, default=42,
                    help="Data seed: selects WHICH images land in each split (independent of "
                         "the model/training seed). Run once per data seed used in the campaign "
                         "(e.g. 42, 43, 44). Seed 42 writes the historical filename.")
    args = ap.parse_args()

    print("--- Stratified Split Generation ---\n")
    generate_tiny_imagenet_stratified_splits(seed=args.data_seed)
    print("All splits generated successfully.\n")
