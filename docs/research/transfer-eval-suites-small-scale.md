# Transfer-Eval Suites for a 64x64 From-Scratch ViT-Tiny — Survey

*Research-agent report, 2026-06-06. Web-verified against primary sources. Addresses the "why CIFAR-10? that's arbitrary" critique by surveying whether a credible VTAB-style multi-task transfer suite exists at this project's scale (ViT-Tiny/8, 64x64, SSL-pretrained on 1k–100k Tiny-ImageNet images, single RTX 5070 Ti). The group-shared CIFAR-10 linear probe stays; this is about an individual extension on top.*

*Tags: [established] = stated/measured in a primary source; [folklore] = common practice without published justification; [open] = no evidence found. Caveat: the dataset-fact sources below are official (torchvision/TFDS/dataset papers); the VTAB protocol details are from the arXiv PDF + the official google-research page, cross-checked.*

## Q1. What VTAB / VTAB-1k actually is

**[established]** VTAB (Zhai et al. 2019, arXiv:1910.04867) = 19 classification tasks, each framed as classification, in three groups. The protocol: solve every task with **1000 training examples** ("VTAB-1k") and **no task-specific logic**. The 19 tasks (verified against the paper / google-research.github.io/task_adaptation and TFDS):

- **Natural (7):** Caltech101, CIFAR-100, DTD, Flowers102, Oxford-IIIT Pets, SVHN, SUN397.
- **Specialized (4):** EuroSAT, RESISC45, PatchCamelyon, Diabetic Retinopathy.
- **Structured (8):** Clevr-count, Clevr-distance, dSprites-location, dSprites-orientation, SmallNORB-azimuth, SmallNORB-elevation, DMLab, KITTI-distance.

**Linear branch: confirmed [established].** VTAB defines two evaluation modes. The headline mode is **fine-tuning** (1000 examples, swept schedules). It *also* specifies a restricted **"lightweight"** sweep — batch 512, SGD momentum 0.9, LR ∈ {0.1, 0.01}, schedule decays at 1/3 and 2/3, duration ∈ {2500, 10000} steps — and a **linear ("frozen feature") evaluation** variant for cheap comparison. So a linear-probe reading of VTAB is *sanctioned by the benchmark itself*, not improvised (arXiv:1910.04867, §3 protocol + appendix implementation details).

**Resolution [established]:** VTAB resizes **all images to 224×224** (128×128 only for generative-model evals). So canonical VTAB is a 224px benchmark. **There is no published VTAB-at-64px or VTAB-on-ViT-Tiny precedent found [open].** Running VTAB tasks at 64px is therefore a *deviation you own*, not a cited protocol — defensible (you re-use VTAB's task taxonomy and 1k-example spirit) but must be disclosed as a re-scaled variant.

## Q2. What SSL papers actually use for multi-task transfer

Two distinct conventions, both linear-friendly:

- **The "classic ~12-dataset linear-transfer suite"** [established]: Food-101, CIFAR-10, CIFAR-100, Birdsnap, SUN397, Stanford Cars, FGVC-Aircraft, PASCAL VOC2007, DTD, Oxford-IIIT Pets, Caltech-101, Oxford-102-Flowers. Originates in **SimCLR** (Chen 2020, arXiv:2002.05709, §B.8 / Table B.8): both a **linear** branch (L2-regularized multinomial logistic regression / LBFGS on frozen features, no augmentation) **and** a **fine-tune** branch are reported. **BYOL** (Grill 2020, arXiv:2006.07733, §B / Table) reports the same 12-dataset suite with **both linear and fine-tune** columns. **Barlow Twins** (Zbontar 2021, arXiv:2103.03230, §4.2 "Transfer Learning", Table 6) reports this suite under **linear evaluation** (logistic regression on frozen features) plus VOC07+07 detection — primarily linear, not full fine-tune of all classification tasks.
  - Note: this suite is **all natural images** — it does NOT span VTAB's specialized/structured axes. It answers "is the representation good?" not "does it transfer across *domains*." For the "good variation" argument, VTAB's taxonomy is the stronger frame.
- **Small-data SSL papers:** El-Nouby et al. (arXiv:2112.10740, "Are Large-scale Datasets Necessary…") pretrains *on the target-task data itself* (Stanford Cars, Sketch, COCO, iNaturalist, Flowers, etc.) and finds **denoising/MIM (BEiT-style) far more robust to small pretraining data than embedding-comparison methods (incl. BT-family)** — directly relevant: it is *evidence your BT runs may transfer worse than an MAE teammate's at small N* [established premise]. It does not use a fixed VTAB suite; it uses per-domain fine-tuning. "ViT on small-scale datasets" works (e.g. arXiv:2210.07240, BMVC'22) evaluate on CIFAR-10/100, CINIC-10, SVHN, Tiny-ImageNet, Aircraft, Cars — a small-image-friendly set, but as supervised/from-scratch ViT benchmarks, not SSL transfer suites.

**Verdict:** linear-only transfer tables are *standard and sufficient* for SSL representation claims (SimCLR, BYOL, BT all lead with linear). You do not need fine-tuning to make a credible transfer argument [established].

## Q3. Which candidate tasks survive 64×64 (native res / size / damage)

Resolutions and sizes verified against torchvision docs / TFDS / dataset papers.

| Task | Native res | Classes | Test size | 64px verdict |
|---|---|---|---|---|
| CIFAR-10 / -100 | 32 (up) | 10 / 100 | 10k / 10k | Fine — upscaled, no loss [established 32px] |
| SVHN | 32 | 10 | 26k | Fine — digit recognition trivially survives [established] |
| **EuroSAT** | **64 NATIVE** | 10 | ~5.4k (no official test split; common 80/20 → ~5.4k) | **Ideal — zero resize artifacts** (Helber arXiv:1709.00029; 27k imgs total, 64×64, RGB version) [established] |
| GTSRB | variable (~15–250px, mostly small) | 43 | 12.6k | OK — signs survive 64px; mild loss [established small native] |
| STL-10 | 96 | 10 | 8k | OK — mild downscale, 96→64 [established] |
| RESISC45 | 256 | 45 | ~6.3k (700/class, split-dependent) | Survives but lossy — scene gist > fine texture; 256→64 loses detail but task is coarse-scene [established 256px] |
| **PatchCamelyon** | **96** | 2 | 32,768 | Survives as *harder*; metastasis cues are cellular — 96→64 risks degrading the subtle signal. Binary, huge test set (low probe noise) [established 96px, 327,680 total] |
| DTD | variable (300–640px) | 47 | ~1,880 (40/class, split 1) | Texture survives downscale reasonably; small test → noisier probe [established] |
| Oxford-IIIT Pets | variable (~few hundred px) | 37 | 3,669 | Fine-grained-ish; 64px hurts breed cues, likely *harder not degenerate* [established] |
| Flowers102 | variable | 102 | 6,149 | Color/shape survive; fine detail lost; many classes / small per-class → noisy [established] |
| **FGVC-Aircraft** | variable (hi-res) | 100 (variant) | 3,333 | **Likely destroyed at 64px** — distinguishing variants needs fine markings/text; expect near-floor. Avoid / use as a deliberate negative [folklore but well-grounded] |
| Stanford Cars | variable | 196 | 8,041 | Same risk as Aircraft — fine-grained, 64px likely degenerate. Avoid. |
| Clevr-count | 480×320 (synthetic) | counts (≤10 → ~8-way) | 15k | Synthetic, high-contrast objects — **counting plausibly survives 64px** (large objects); easiest structured task per VTAB [established "Clevr-count very easy"] |
| dSprites (loc/orient) | 64 NATIVE | binned | ~large | **64px native** — survives perfectly *as pixels*; but predicting orientation of a tiny white sprite at 64px is the hard part regardless of resize [established 64px] |
| SmallNORB (azimuth/elev) | 96 (grayscale) | binned | 24,300 | Grayscale toys; 96→64 fine; pose regression hard for any frozen feat |
| DMLab | rendered frames | distance bins | ~22k | Survives res; semantically hard |

**Degenerate-at-64px flags [folklore, well-grounded]:** FGVC-Aircraft and Stanford Cars — fine-grained ID needs detail 64px removes; say so explicitly and exclude (or include one *only* to demonstrate the resolution limit). Birdsnap (defunct download, links rot) — avoid [open/practical]. SUN397 (108k imgs, 397 classes) — heavy, scene-gist survives but overkill.

## Q4. Protocol cost (5.7M ViT-Tiny @ 64px, RTX 5070 Ti 16GB)

- **(a) Linear probe on cached frozen features [recommended].** Forward-pass each task's train+test once → cache 192-d (CLS) or pooled features → logistic regression (sklearn / LBFGS) or a 1-layer torch head. Per task: one forward pass over ≤~30k images (<1 min on the GPU) + a CPU/GPU logistic fit (seconds–minutes). **Whole 5-task suite per checkpoint: minutes, effectively free.** This is exactly SimCLR/BT's protocol. With 8 split sizes × N seeds, still trivial.
- **(b) k-NN probe [free, zero training].** Cache features, cosine k-NN (DINO-style, arXiv:2104.14294, §5). Cheapest of all, no fit, no probe-hyperparameter sweep. Good as a *sanity companion* to (a); slightly lower numbers, very low variance.
- **(c) VTAB-1k-style fine-tuning [skip].** 1000 ex/task, 2500–10000 steps, but full backprop through 5.7M params per task per checkpoint. At 64px a step is cheap, but ×(tasks × checkpoints × seeds) it dwarfs the near-free probe and adds optimizer-hyperparameter confounds. Not needed for a credible claim (Q2). Reserve for *one* task if you want a fine-tune data point, mirroring the planned CIFAR-10 PEFT follow-up.

**Sufficiency [established]:** linear-only transfer is the published norm for SSL representation quality (SimCLR §B.8, BYOL §B, BT §4.2). k-NN-only is accepted as a fast proxy (DINO §5). You are on solid ground reporting linear (+optional k-NN), no fine-tuning.

## Q5. Recommendation — a 5-task, three-axis, 64px-safe, linear-probe suite

Keep group CIFAR-10 as-is. Add this **individual** suite (all torchvision-native, all cached-feature linear probe):

1. **CIFAR-100** (natural, fine-grained-ish, 100-way) — natural axis, harder than CIFAR-10, 32px (no resize loss), torchvision. Grounded in classic SSL suite + VTAB-natural.
2. **SVHN** (natural/text-ish, 10-way, 32px) — different natural distribution (digits, low semantic overlap with Tiny-ImageNet), large test set → low probe noise. VTAB-natural, torchvision.
3. **EuroSAT** (specialized — satellite, 10-way, **64px NATIVE**) — the single best fit: VTAB-specialized axis, zero resize artifacts, small/fast, torchvision. The strongest "different domain" datapoint.
4. **PatchCamelyon / PCAM** (specialized — medical, binary, 96→64px) — second specialized axis (medical vs satellite), huge test set (32,768) → near-zero probe variance, torchvision. Disclose the 96→64 detail loss.
5. **Clevr-count** (structured — counting, ~8-way, synthetic) — the structured axis via VTAB's *easiest* structured task (least likely to be at floor); large synthetic objects survive 64px. Available via TFDS / HF (not torchvision — minor friction; verify loader). If loading is painful, substitute **dSprites** (64px native, HF) but expect lower scores.

This spans **natural / specialized-satellite / specialized-medical / structured** — exactly VTAB's three-category "good variation" argument — in 5 cheap linear probes. All survive 64px (EuroSAT/dSprites natively; CIFAR-100/SVHN are 32px; PCAM/Clevr degrade gracefully).

**Grounded vs improvised:**
- *Grounded* [established]: linear-probe-on-frozen-features protocol (SimCLR/BYOL/BT); the VTAB three-category taxonomy and 1k-example spirit; CIFAR-100/SVHN/DTD/EuroSAT/PCAM as VTAB members; linear-only being sufficient for SSL claims.
- *Improvised (disclose)* [open]: running any of this **at 64px on a ViT-Tiny** — no published VTAB-at-64px / tiny-backbone precedent; the specific 5-task subset; treating Clevr-count as the structured representative.

**Main threats to validity:**
- **Floor effects on structured/fine-grained tasks** [established direction, untested at this exact scale]. VTAB itself reports autoencoders/some SSL *below from-scratch* on structured tasks, and structured tasks are where frozen features are weakest (arXiv:1910.04867). A from-scratch BT ViT-Tiny on ≤100k 64px images may sit near chance on dSprites/SmallNORB/DMLab → those readings carry no signal. Clevr-count is the safest structured pick; still, report chance levels alongside accuracy and be ready to interpret "near floor" as itself a finding (small-data SSL doesn't reach structured tasks).
- **CLAUDE.md's own warning:** the project conventions explicitly note specialized VTAB tasks (medical/satellite) "won't share features with the pre-training corpus" (Tiny-ImageNet). That is *the point* of the extension (does the rep transfer out-of-domain?), but it predicts low EuroSAT/PCAM numbers — frame as a transfer-gap measurement, not a failure.
- **Resolution confound:** at 64px you cannot separate "weak representation" from "task needs detail 64px removed" (Aircraft/Cars/PCAM). Stick to tasks where 64px is merely harder, not degenerate; exclude Aircraft/Cars.
- **Probe noise on small test sets** [established]: DTD (~1,880), Pets (3,669), Flowers (6,149/102-way) give noisy estimates at small N — prefer large-test tasks (SVHN, PCAM, EuroSAT) and report seeds.
- **El-Nouby caveat** [established, arXiv:2112.10740]: embedding-comparison SSL (BT-family) is empirically the *least* small-data-robust class; expect your transfer numbers to lag MAE/JEPA teammates — anticipate this in the write-up rather than be surprised.

## Sources
- VTAB: arXiv:1910.04867; https://google-research.github.io/task_adaptation/ ; https://github.com/google-research/task_adaptation ; TFDS catalog (eurosat, resisc45, patch_camelyon, clevr, dsprites, smallnorb).
- SSL suites: SimCLR arXiv:2002.05709 (§B.8); BYOL arXiv:2006.07733 (§B); Barlow Twins arXiv:2103.03230 (§4.2, Table 6); DINO k-NN arXiv:2104.14294 (§5); El-Nouby arXiv:2112.10740; small-ViT arXiv:2210.07240.
- Dataset facts: EuroSAT arXiv:1709.00029 (64px native, 27k, 10 cls); PCAM github.com/basveeling/pcam + torchvision PCAM docs (96px, 327,680, binary, 32,768 test); RESISC45 TFDS (256px, 31,500, 45 cls); torchvision datasets (StanfordCars, SVHN, Flowers102, DTD, GTSRB, PCAM, FGVCAircraft, EuroSAT confirmed present).
