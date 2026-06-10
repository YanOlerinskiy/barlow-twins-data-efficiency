# Power-of-2 Batch Sizes — Hardware Requirement or Folklore?

*Research-agent report, 2026-06-09. Web-verified against primary sources (NVIDIA Deep Learning Performance docs, CUTLASS docs, empirical myth-busting benchmarks). Verdicts: **[established]** = stated/measured in a primary/authoritative source; **[folklore]** = widespread practice with no published proof; **[inference]** = my reasoning from established facts; **[open]** = no evidence found. Concrete question: a student training **ViT-Tiny/8 (hidden 192, ~65 tokens/img), TF32, single RTX 5070 Ti (Blackwell sm_120)** wants to switch pre-training batch from **256 → 250** so dataset splits divide evenly. Does 250 carry a real penalty? Bottom line up front: **no measurable penalty; the power-of-2 rule is folklore, the real rule is "multiple of a small number (4/8/16)", and even that barely touches the batch dimension here.** See `validation-eval-cadence-best-practice.md` for the broader thesis-methods context.*

## Q1. Where the convention comes from — real reasons vs cargo cult

Four candidate justifications; only one is a real (and weak) constraint, and **none requires a power of 2** — at most a multiple of a small number:

- **(a) Memory alignment / coalesced access** — real at the level of the *innermost contiguous dimension* (alignment to 16/32/128-byte transactions, warp-coalesced loads), but the **batch dimension is the outermost stride** of an `[B, C, H, W]` / `[B, T, D]` tensor, so its size does not affect intra-row coalescing. Alignment cares about the feature/channel dim, not B [inference from CUDA memory model; NVIDIA GPU Performance Background]. Does **not** require power-of-2.
- **(b) Warp / thread-block sizing (warp = 32)** — warps are a kernel-launch detail the GEMM library chooses; the user-visible tensor dimensions need not be multiples of 32. Modern kernels handle arbitrary M with a remainder (predicated) tile [established, CUTLASS *Efficient GEMM*]. Does **not** require power-of-2.
- **(c) Tensor-Core tile / alignment** — the *one real* constraint, and it is **divisibility by 4/8/16 (per precision), not a power of 2** (Q2). Tile/wave quantization (Q2) are second-order throughput effects, again about divisibility by the *tile size*, not powers of 2.
- **(d) Allocator bucketing** — the PyTorch caching allocator rounds block requests up to internal size classes (512-byte multiples; large blocks rounded to 2 MB granularity), so a "rounder" batch does not get a special bucket. Power-of-2 helps the allocator only in the trivial sense that *any* fixed batch size reuses the same blocks; 250 reuses just as well as 256 once steady-state [established, PyTorch CUDACachingAllocator design; the "rounder = better" claim is **folklore**].

**Net**: the power-of-2 habit is mostly cargo-cult inherited from an era of hand-tuned kernels and from conflating "power of 2" with "multiple of 8" [folklore]. The defensible kernel rule is *multiple of 8/16*, not *power of 2* [established, Q2].

## Q2. Tensor-Core alignment — the actual constraint, and exact multiples

NVIDIA's *Matrix Multiplication Background* and *GPU Performance / Linear-FC* guides give per-precision divisibility targets for the GEMM dimensions M, N, K (not the tensor as a whole):

- **FP16: multiples of 8** (64 on A100 for best efficiency) [established].
- **TF32: multiples of 4** (32 on A100 for best efficiency) [established]. ← relevant precision here.
- **INT8: multiples of 16** (128 on A100) [established].
- **FP64: multiples of 2** (16 on A100) [established].
- *Best-efficiency tiling note*: when dimensions are small, NVIDIA suggests divisible by **≥64, ideally 256**, to "streamline tiling and reduce overhead" [established, *Linear/FC Layers* guide]. This is the strongest form of the "round numbers" guidance — and it is about reducing **tile-quantization overhead**, still not powers of 2.

**Is it power-of-2 or divisibility by 8/16?** Unambiguously the latter: NVIDIA never states "power of 2." 256 satisfies the rule because 256 is a multiple of 8/4/16, *not* because it is 2^8. **248 (=8×31) is exactly as Tensor-Core-friendly as 256** [established].

**Where does BATCH enter M/N/K?** For a linear layer (NVIDIA *Linear/FC* doc) [established]:

| Phase | M | N | K |
|---|---|---|---|
| Forward | outputs | **batch** | inputs |
| Activation-grad | inputs | **batch** | outputs |
| Weight-grad | inputs | outputs | **batch** |

So batch is the **N dim** (fwd/dgrad) and the **K dim** (wgrad). For a ViT the GEMMs in the attention/MLP blocks fold tokens into the "row" count: effective **M = batch × tokens** (a `[B·T, D]` matrix), and the per-layer weight GEMMs have N/K = hidden dims (192, 768). **Tile/wave quantization is dominated by the feature/hidden dims (N, K) and by M = B·T, not by B alone** [inference from the GEMM layout; consistent with NVIDIA's worked examples being driven by N/K]. With T≈65, M = 250·65 = 16250 vs 256·65 = 16640 — both huge relative to any tile (typ. 64–256), so the **remainder tile is a sub-percent fraction of total tiles either way** [inference].

- **Tile quantization** [established]: when a dim is not divisible by the thread-block tile, boundary tiles do wasted work; NVIDIA's example `8192×136×8192` with 256×128 tiles does ~1.5× extra ops — but that bite is large only when the dim is *small and comparable to the tile size*. For M≈16k it is negligible.
- **Wave quantization** [established]: total tiles must fill whole "waves" across the SMs; a near-miss can halve GFLOPS in NVIDIA's `N: 1536→1544` example. This is driven by the *total tile count* (mostly N, K and SM count), and crucially is **non-monotonic** — a power of 2 is not protective and can even land on a bad wave boundary [established, *Matrix Multiplication Background*].

## Q3. Empirical evidence — power-of-2 buys essentially nothing

- **Raschka / Lightning AI, "No, We Don't Have to Choose Batch Sizes as Powers of 2" (2022)** [established, measured]: training a model at batches near 128 and near 512: **128→9.78 min, 127→9.80, 129→9.92, 156→9.38, 100→10.50**; near saturation **511→8.74, 512→8.71, 513→8.72**; 4×V100 **255→2.95, 256→2.87, 257→2.86** (257 *beat* 256). Conclusion: differences "barely noticeable," power-of-2 has no runtime advantage; the better guideline is multiples of 8 for Tensor-Core *activation*, not powers of 2.
- **W&B "Do Batch Sizes Actually Need To Be Powers of 2?"** [established, measured]: P100 (no Tensor Cores) ResNet 128 vs 129 both 26 min; DeiT 64 vs 65 both 39 min; ConvNeXt 64 vs 65 both 37 min; GPU util within noise. RTX A4000 (Tensor Cores) ResNet50: same 16 min for both sizes. "No sudden breaks at specific batch sizes."
- **Magnitudes to quote honestly**: non-power-of-2 **vs power-of-2 ≈ 0%** [established]. Non-multiple-of-8 vs multiple-of-8 in a *Tensor-Core-bound, small-dimension* GEMM can cost **single-digit %** in the worst case (and via vocab/hidden-dim padding the *feature* dim can matter a lot — the well-known "pad vocab to ×8" result) [established, NVIDIA + HF tokenizer guidance]. Crucially, **that penalty attaches to the feature/hidden dims, not the batch dim**, and only when those dims are small. The strong evidence is for "multiples of 8 help on small feature dims"; evidence that *batch-size* power-of-2 matters is **absent / debunked** [established that it's debunked].

## Q4. 250 vs 256 specifically, for this ViT-Tiny / TF32 / Blackwell setup

- 250 = 2·5³ (250/8 = 31.25, *not* a multiple of 8; 250/4 = 62.5, not a multiple of 4 either). 256 = 2⁸ = ×4, ×8, ×16, ×32. So under TF32 the relevant rule (**multiple of 4**) is satisfied by 256 and **248/252** but **not by 250** [established].
- Does this matter? **Almost certainly not, for three compounding reasons** [inference, strongly supported]:
  1. **Batch is M (=B·T) or N/K, and is large.** M = 250×65 ≈ 16k. The GEMM kernel pads/predicates the remainder row-tile; the wasted fraction is `tile/M` ≈ (64–256)/16k = **0.4–1.6% at most, and only on the last partial tile**, identical in spirit whether B is 250 or 256. The hidden dims (192, 768) — which *are* fixed multiples of consistent numbers — dominate quantization, and they don't change with batch [inference, NVIDIA GEMM layout].
  2. **TF32 only needs multiples of 4**, the weakest of all the rules; the penalty for missing it is the gentlest [established].
  3. **Tile quantization rounds M up to the tile size regardless** — 250 and 256 both round up to the next tile boundary (e.g. both into the same count of 64- or 128-row tiles, or differ by at most one partial tile), so the *executed* work is near-identical [inference, CUTLASS tile dispatch].
- **Is 250 worse than 248 (×8) or 256 (×256)?** Theoretically 248/256 align to the TF32 ×4 rule and 250 does not, but the realized difference on M≈16k is **expected to be well under 1%, i.e. within run-to-run noise** [inference; consistent with Raschka's 127/129/100 spread of ≤1 min on minutes-long runs]. No published benchmark shows a 250-class case losing measurably. **[open]** for the *exact* Blackwell/TF32/ViT-Tiny number — but the mechanism predicts negligible.

## Q5. Other considerations

- **(a) Distributed / multi-GPU sharding**: power-of-2 (more precisely, divisibility by the world size and by the micro-batch count) eases even global-batch splitting and gradient-accumulation arithmetic [established convention]. **Single GPU here → N/A.** Pick any B; no sharding constraint [inference]. (Even multi-GPU only needs divisibility by #GPUs, not a power of 2.)
- **(b) Convergence / generalization**: **no evidence that power-of-2 batch sizes help optimization or generalization** [established absence]. The batch-size↔generalization literature (Keskar et al. 2017 sharp-vs-flat minima arXiv:1609.04836; Hoffer et al. "Train longer, generalize better" arXiv:1705.08741; Smith & Le 2017) is entirely about the *magnitude* of the batch (large→sharper minima, etc.), never its 2-adic valuation. The claim "powers of 2 train/generalize better" is pure **[folklore]**. A 2.3% change (256→250) in batch magnitude has a negligible optimization effect and is dwarfed by LR/epoch choices [inference].
- **(c) Allocator fragmentation**: a fixed batch (any value) reaches allocator steady state after the first few steps and reuses cached blocks; 250 fragments no more than 256 [established, PyTorch caching-allocator design]. The only real memory point is that 256 uses *slightly more* VRAM than 250 (linear in B) — favoring 250 if anything [inference].

## Q6. Bottom line for the student

- **Switching 256 → 250 will not cost measurable throughput** on ViT-Tiny/TF32/single Blackwell. Expected difference: **sub-1%, inside run-to-run variance** [inference, strongly supported by Q2–Q4]. The power-of-2 rule is **folklore**; the real hardware rule is *multiple of 8 (FP16) / 4 (TF32) / 16 (INT8)* and it bites the **feature/hidden dims** (here fixed) far more than the **batch dim** (here large, so it's nearly all padded/predicated away) [established].
- **Defensible methods-section statement**: *"Batch size 250 was chosen so all dataset splits divide evenly. Although powers of two are conventional, NVIDIA's Tensor-Core guidance specifies divisibility by 4 (TF32), not powers of two, and that constraint primarily governs the feature dimensions, which are fixed by the ViT-Tiny architecture; the batch dimension enters the GEMMs multiplied by the token count (M ≈ B·T ≈ 16k), so any tile-quantization difference between 250 and 256 is a sub-percent, sub-noise effect (cf. Raschka 2022; NVIDIA Matrix-Multiplication Background)."*
- **Cleanest alternative if you want divisibility AND multiple-of-8/4**: **batch 200** (= 8×25 = 4×50; divides 1k/2k/4k/8k/16k/32k/64k/100k *exactly*, and the 200-divisible splits are also the perfectly class-balanced ones per `CLAUDE.md`). 200 satisfies the TF32 ×4 and FP16 ×8 rules and your stratification simultaneously. **248** also works (×8) if you want to stay near 256 but it doesn't divide the splits. If you ever switch to FP16/BF16, prefer 200 or 248 over 250; under TF32 even this is academic.
- **Verdict**: go with 250 if it's the cleanest fit; **200 is the marginally "correct-by-the-book" choice** that costs you nothing and silences any reviewer. Do not lose sleep over either — and do **not** claim a *measured* hardware benefit; cite the convention-vs-rule distinction.

## Sources
- NVIDIA, *Matrix Multiplication Background User's Guide* — per-precision multiples (FP16 ×8/64, TF32 ×4/32, INT8 ×16/128, FP64 ×2/16); tile quantization (8192×136×8192 → ~1.5× ops) and wave quantization (N 1536→1544 ~½ GFLOPS). https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
- NVIDIA, *Optimizing Linear/Fully-Connected Layers* — batch↔M/N/K table (batch = N in fwd/dgrad, K in wgrad); "divisible by 4 (TF32)/8 (FP16)/16 (INT8)", "≥64 ideally 256" for small dims. https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
- NVIDIA, *Train With Mixed Precision* / *GPU Performance Background*. https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
- NVIDIA, *Tips for Optimizing GPU Performance Using Tensor Cores* (divisible-by-8 FP16 / 16 INT8 for Tensor-Core activation). https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
- CUTLASS docs — *Efficient GEMM in CUDA* / GEMM API (predicated remainder tiles; tile-size dispatch by problem size). https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- S. Raschka (Lightning AI), *No, We Don't Have to Choose Batch Sizes as Powers of 2* (2022) — measured 127/128/129/156/100, 511/512/513, 255/256/257. https://sebastianraschka.com/blog/2022/batch-size-2.html
- W&B, *Do Batch Sizes Actually Need To Be Powers of 2?* — P100/A4000 ResNet/DeiT/ConvNeXt, no breaks. https://wandb.ai/datenzauberai/Batch-Size-Testing/reports/Do-Batch-Sizes-Actually-Need-To-Be-Powers-of-2---VmlldzoyMDkwNDQx
- Batch-size↔generalization (magnitude, not power-of-2): Keskar et al. arXiv:1609.04836; Hoffer et al. arXiv:1705.08741; Smith & Le 2017.
- PyTorch CUDA caching allocator (size-class rounding, not power-of-2 buckets) — PyTorch `CUDACachingAllocator` / docs.
- Prior report: `validation-eval-cadence-best-practice.md` (thesis methods context); `CLAUDE.md` (split sizes, 200-divisibility).
