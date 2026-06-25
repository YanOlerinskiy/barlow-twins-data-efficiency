# Reducing Data in Visual AI

A TU Delft research project (2025/2026) part of the **Research Project (CSE3000) Course** at TU Delft. This project investigates the impact of data reduction on self-supervised learning and data-efficiency for visual AI foundation models.

**Team Members:**
- Maksim Plotnikov
- Dimo Terziev
- Yan Olerinskiy
- Leonid Margulis
- Makar Kuleshov

**Responsible Professor:**
- Jan van Gemert

**Supervisors:**
- Petter Reijalt
- Alex Manolache

## Background and motivation

Data is powering AI. Most data, however, is in the hands of LargeCompany^TM, creating a privileged few that have huge data, and a long tail of universities, SMEs and researchers, that have limited access. Moreover, large dataset are hard to process and curate, making it difficult to control for fairness, copyright, and data biases.

In this project we will explore how to more efficiently use data to train visual AI. Specifically, we will investigate the effect of the amount of data on current visual foundation models typically a vision transformer (VIT) trained self-supervised which is fine-tuned by PEFT on down-stream tasks.

In this project we will evaluate how to scale-down the huge data problem, to a manageable, but still
representative, setting where we can tackle the underlying research problem, without being constrained by
huge compute.

## Research Questions for the Sub-Projects

**Whole group:**
The research question for the whole group is how to evaluate data efficiency by learning curves for small-compute self-supervised pre-trained visual foundation models, which are then evaluated on a set of down-stream tasks.

**For each of the 5 sub-projects:**
The research question for each individual student, is to investigate, implement, and critically evaluate a popular self-supervision method on data-efficiency. Each student implements one of the following methods:

- **Barlow Twins** - Redundancy reduction principle for self-supervised learning
- **DINO** - Vision Transformers with self-supervised knowledge distillation
- **MoCo** - Momentum Contrast for unsupervised visual representation learning
- **JEPA** - Joint Embedding Predictive Architecture
- **MAE** - Masked Autoencoders for self-supervised learning on vision

Each method is compared and evaluated across different data scales to assess data efficiency.

## Project Structure

```
├── data/
│   ├── dataset_utils.py      # Data loading and preprocessing utilities
│   └── generate_splits.py    # Script to generate dataset splits
├── models/                  # Implementations of self-supervised methods
│   ├── barlow_twins.py      # Barlow Twins implementation
│   ├── dino.py              # DINO implementation
│   ├── moco.py              # MoCo implementation
│   ├── jepa.py              # JEPA implementation
│   └── mae.py               # MAE implementation               
├── splits/                  # Pre-computed dataset indices for each split
│   ├── tiny_imagenet_1k_indices.json
│   ├── tiny_imagenet_2k_indices.json
│   ├── tiny_imagenet_4k_indices.json
│   ├── tiny_imagenet_8k_indices.json
│   ├── tiny_imagenet_16k_indices.json
│   ├── tiny_imagenet_32k_indices.json
│   └── tiny_imagenet_64k_indices.json
├── shared_config.py         # Shared configuration (model params, transforms)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Data

### Pre-training Dataset

The project uses the **Tiny ImageNet** dataset for pre-training:
- **Size**: 100,000 training images (when full dataset is used)
- **Resolution**: 64×64 pixels
- **Classes**: 200 object categories
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/Maysee/tiny-imagenet)

### Available Splits

Training data is available at multiple scales to study data efficiency: **1K, 2K, 4K, 8K, 16K, 32K, 64K and 100K** images.

Split indices are pre-computed and stored as JSON files in the `splits/` directory for reproducibility.

## Models

All sub-projects use the same **Vision Transformer (ViT) - Tiny / 8** backbone, trained with different self-supervised learning methods:

### Shared Architecture: Vision Transformer (ViT) - Tiny / 8

**Model Details**:
- Patch size 8×8 → 64 patches per image
- Embedding dimension: 192
- MLP Hidden: 768
- 12 transformer layers
- 3 attention heads
- 5.7M params
- 1.3 FLOPs

### Self-Supervised Learning Methods

| Method | Researcher |
|--------|---|
| **Barlow Twins** | Yan Olerinskiy |
| **DINO** | Leonid Margulis |
| **MoCo** | Makar Kuleshov |
| **JEPA** | Maksim Plotnikov |
| **MAE** | Dimo Terziev |

Each method implements a different approach to self-supervised pre-training, which are then evaluated on their data efficiency across the dataset splits.

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/DDTerziev04/reducing-data-in-visual-ai.git
cd reducing-data-in-visual-ai
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
