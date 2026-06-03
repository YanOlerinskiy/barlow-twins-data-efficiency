# CLAUDE.md

Context for a CLI coding agent working on this research project.

## What this project is

CSE3000 Research Project at TU Delft, Q4 2025/26. Topic: **data efficiency of self-supervised learning (SSL) methods for visual foundation models in a small-compute regime.**

The project runs as a group of 5 students sharing one experimental pipeline. Each student owns one SSL method, and the group produces a single aggregated learning curve comparing all five. The group-level question is *how the methods compare* on data efficiency (not "how to evaluate it" — that part is settled by the shared learning-curve protocol). Individual sub-questions should be framed concretely as *"how can $X$ be optimised for data efficiency?"* and refined as the work matures (e.g.\ "which augmentations prevent feature collapse at small data scales?").

- **My method**: Barlow Twins (Zbontar et al., 2021)
- **Teammates' methods**: DINO (Leonid Margulis), MAE (Dimo Terziev), MoCo (Makar Kuleshov), JEPA (Maksim Plotnikov)
- **Owner**: Yan Olerinskiy
- **Responsible professor**: Jan van Gemert
- **Supervisors**: Petter Reijalt, Alex Manolache

## The experimental pipeline

Shared by the whole group:

- **Backbone**: ViT-Tiny/8 (~5.7M params, 192-dim embeddings, 8×8 patches on 64×64 inputs → 64 tokens per image, 12 layers, 3 heads, MLP hidden 768). Exact config in `reducing-data-in-visual-ai/shared_config.py` as `VIT_TINY_8_KWARGS`.
- **Pre-training dataset**: Tiny ImageNet (100k images, 64×64, 200 classes, HF: `Maysee/tiny-imagenet`).
- **Pre-training split sizes** (the x-axis of the learning curve): {1k, 2k, 4k, 8k, 16k, 32k, 64k, 100k} images. The splits are **nested** and **stratified** by class — they come from a single master shuffled index list (`tiny_imagenet_shuffled_indices.json`, seed 42), so smaller splits are prefixes of larger ones and every prefix size divisible by 200 is perfectly class-balanced.
- **Downstream task**: CIFAR-10, upsampled from 32×32 to 64×64.
- **Downstream protocol**: **linear probing initially** (freeze the pre-trained encoder, train only a linear classifier on top). PEFT fine-tuning is a planned follow-up if time allows.
- **Results format**: CSV with columns `method, fraction, seed, accuracy`.

Per-student deliverable: one learning curve from the shared CSV, then merged into the group plot.

## My specific method: Barlow Twins

- **Loss**: cross-correlation matrix between two augmented views' projector outputs, pushed toward the identity. Invariance term (diagonal → 1) + redundancy reduction term (off-diagonal → 0), weighted by λ.
- **Reference implementation to start from**: `solo-learn` (https://github.com/vturrisi/solo-learn). It has a ViT recipe; adapt their config rather than starting from scratch.
- **Original paper used**: ResNet-50, ImageNet, batch 2048, 1000 epochs, projector 8192-dim, LARS, λ = 5e-3.
- **Our scaled-down setup**: ViT-Tiny/8, Tiny ImageNet subsets, batch ~256-512, ~100-300 epochs with early stopping, projector ~1024-d, AdamW, λ scaled to projector dim (~1e-2 range).
- **Known risks**: ViT instability under SSL (MoCo v3 documented it; standard fix is frozen patch-embed for early epochs, low LR, gradient clipping). solo-learn's ViT recipe handles this.
- **Likely extension direction**: the original Barlow Twins paper notes openness to richer augmentations — exploring augmentation choices is a natural way to refine the individual sub-question into something concrete (e.g.\ "which augmentations let Barlow Twins avoid feature collapse at the smallest splits?").

## Tools

- **Framework**: PyTorch
- **Models / data**: Hugging Face `transformers`, `datasets`, `peft`
- **SSL recipes**: `solo-learn` as the primary reference; `lightly` as a fallback
- **Experiment tracking**: TBD (likely Weights & Biases or TensorBoard)
- **Compute**: local GPU by default. Cluster access exists (Makar has it with low priority); DAIC (https://daic.tudelft.nl) is available as a fallback.

## What's in this workspace

```
docs/
  CLAUDE.md                                this file
  project_description.md                   original project brief from supervisors
  research_plan.tex                        submitted Q4 research plan
meeting_notes/
  meeting1.md … meeting3.md                notes from supervisor meetings
reducing-data-in-visual-ai/                shared group repository (sibling clone)
  data/
    dataset_utils.py                       ViTDatasetWrapper + get_pretraining_dataloader(split, ...)
    generate_splits.py                     stratified shuffled master index list (seed 42)
    tiny_imagenet_shuffled_indices.json    canonical master order — all five students use this
  models/
    barlow_twins.py                        my implementation (to be added)
    dino.py / moco.py / jepa.py / mae.py   teammates' implementations
  splits/                                  pre-computed per-size index files (1k…64k)
  shared_config.py                         VIT_TINY_8_KWARGS, tiny_imagenet_transform
  requirements.txt
  README.md
```

The `reducing-data-in-visual-ai/` folder is the **shared group repository**. Anything that goes in there is co-owned by the five group members and must stay compatible with everyone's pipeline. The data loader, splits file, backbone config, downstream evaluation, and shared config all live there.

**Shared dataloader API** (the one piece of shared code agents are most likely to interact with):

```python
from data.dataset_utils import get_pretraining_dataloader
loader = get_pretraining_dataloader(split=8000, batch_size=256, num_workers=4)
# loader yields {'pixel_values': tensor[B, 3, 64, 64], 'label': tensor[B]}
```

**Next coding tasks for me**: add `models/barlow_twins.py` (Barlow Twins loss + projection head + training loop wired to `get_pretraining_dataloader`), smoke-test at `split=1000`, then sweep the rest of the splits.

The loader uses the first `split` indices of `tiny_imagenet_shuffled_indices.json`, so subsets are nested and class-stratified — `split=1000` is a strict subset of `split=8000`, and any split size divisible by 200 is perfectly class-balanced.

## Working conventions

- **Don't reimplement methods from scratch.** Start from solo-learn / lightly / the authors' original repo and adapt.
- **Don't go broad on hyperparameters.** Per supervisor advice, one sensible configuration is enough; only revisit specific hyperparameters if a clear issue appears. If a paper uses batch size 64, try 32/128, not more.
- **Use the shared data loader.** Anything that bypasses it produces incomparable results.
- **Keep code reproducible.** Fixed seeds for both the data subset and the training run, recorded alongside results.
- **Write results in the shared CSV format**, not custom per-experiment formats.
- **Pre-training/fine-tuning domain match.** Pre-training data should be related to fine-tuning data. CIFAR-10 ↔ Tiny ImageNet is fine; specialised VTAB tasks (medical, satellite) won't share features with the pre-training corpus.
- **Time budget per run.** Supervisors flagged that a single pre-training run should stay within a day of wall-clock; ~8 hours is the rough ceiling for "still reasonable". If a config is heading much longer than that, stop and rethink (smaller batch, fewer crops, etc.) before pushing through.
- **Epoch heuristic.** More epochs for smaller datasets and smaller batch sizes (keeps the total number of gradient steps in roughly the same range). The fixed-epoch budget across split sizes is wrong by default.
- **Don't tune for faster training.** Data efficiency is the goal, not throughput. Don't sacrifice the comparison to make individual runs cheaper.
- **Loss going up isn't automatically a bug.** Several methods (notably JEPA) have non-monotonic SSL loss curves that still produce useful representations. Validate with a downstream metric (e.g. kNN every few epochs), not the SSL loss alone.

## Timeline anchors

- **Now**: Week 4 of Q4 (12 May 2026).
- Week 5: midterm presentation (preliminary learning curve expected).
- Week 6: aggregated five-method plot, MVP milestone.
- Weeks 7–9: paper drafts v1 and v2, peer review, responsible-professor feedback.
- Week 10: final paper, poster, final presentation.

## Things to ask me before assuming

- Exact batch size and epoch budget for pre-training (epoch counts should scale up at smaller splits).
- Whether multi-seed runs are in scope yet (default: not yet, baseline curve first).
- Whether the downstream evaluation should add a quick kNN probe every few epochs as a sanity check during pre-training, or only run linear probing at the end.
- Whether to start moving toward cluster compute (Makar / DAIC) or stay local.