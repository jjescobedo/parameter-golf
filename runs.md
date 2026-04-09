# Parameter Golf — Run Log

## Current Best
**val_bpb: 1.3152 | Artifact: ~15.7MB | Config: 14L ConvFFN** (tight on 16MB limit)
**Safe pick: val_bpb: 1.3178 | Artifact: ~14.9MB | Config: 13L ConvFFN**
Config: 13-14L, 512dim, 2x MLP + causal depthwise conv (k=3), sliding window eval (stride=64), int5/6 + zstd-22 + 3% pruning

---

## How We Got Here

### Step 1: Get training working
Got DSMLP cluster access with A30 GPUs. Had to fix PyTorch 2.2 compatibility issues (no bf16 on 1080 Ti, no torch.compile on old PyTorch, uint16 tensor support, MIG device issues). Upgraded to PyTorch 2.4 for torch.compile.

### Step 2: Establish baseline
Trained 9-layer, 512-dim, 2x MLP model with standard eval. val_bpb = 1.3687.

### Step 3: Sliding window eval (free -0.034 bpb)
Added overlapping window evaluation with stride=64. Every token gets ~960 tokens of context instead of random amounts. Pure eval trick, no training change. val_bpb: 1.3687 → 1.3348.

### Step 4: 3x MLP expansion (-0.014 bpb)
Widened MLP hidden dimension from 2x to 3x (1024 → 1536). More expressive feedforward layers. val_bpb: 1.3348 → 1.3211. But artifact grew to 18.7MB (over 16MB limit).

### Step 5: Compression pipeline (-0.007 bpb AND -5.7MB)
Replaced int8+zlib with mixed int5/int6+zstd-22+byte shuffle+3% magnitude pruning. MLP weights get int5 (32 levels), attention gets int6 (64 levels), embeddings stay fp16. val_bpb: 1.3211 → 1.3275 (tiny hit from quantization). Artifact: 18.7MB → 13.0MB.

---

## Experiments Tried

### What Worked
| Technique | Impact | Cost |
|-----------|--------|------|
| Sliding window eval (stride=64) | -0.034 bpb | Zero (eval only) |
| 3x MLP expansion | -0.014 bpb | +4MB before compression |
| Mixed int5/6 quantization | -5.7MB artifact | +0.006 bpb |
| zstd-22 over zlib-9 | ~1MB saved | Zero quality loss |
| Byte shuffle before compression | ~0.5MB saved | Zero quality loss |
| PyTorch 2.4 + torch.compile | 3x faster training | Zero quality loss |
| ConvFFN (causal depthwise conv + 2x MLP) | -0.009 bpb (12L), -0.012 bpb (14L) | More depth from cheaper MLP |

### What Failed
| Technique | Result | Why |
|-----------|--------|-----|
| 10% pruning (vs 3%) | +0.1MB bigger, same bpb | zstd already handles near-zero weights; int5/6 quantization maps small values to 0 anyway |
| TurboQuant 3-bit | val_bpb = 2.61 (destroyed) | Designed for KV cache compression, not weight quantization. 8 levels way too few. |
| TurboQuant adjusted | val_bpb = 1.66 | Still bad. Rotation trick doesn't help when weights already have good distribution from Muon+WD. |
| Kronecker MLP (15 layers) | 1138ms/step (5x slower) | Python for-loops break torch.compile. Tiny matrix multiplies don't saturate GPU. Fewer training steps = worse model. **Dead end for competition.** |
| N-gram complementary training | val_bpb = 1.3614 (worse) | Scaling mean loss doesn't reweight individual tokens. Need per-token loss modification which breaks torch.compile fullgraph. |
| Shared Base + Low-Rank (A4) | NO-GO (all 6/6 matrix types fail) | SVD analysis shows layers share zero structure. Rank-32 captures <70% residual energy for every matrix type. Recommended rank=358 (full rank). Diagonal scaling <10% benefit. Layers are fundamentally different — no shared base possible. |
| Factorized FFN (14L, rank-192) | val_bpb = 1.4033 (worse) | Bottleneck rank loses too much expressivity. 43% slower/step compounds the problem. |
| Soft MoE (9L, 4 experts) | val_bpb = 1.3987 (worse) | Soft routing (all experts, all tokens) gives no specialization. Just redundant features. |
| Butterfly/Monarch MLP (18L) | val_bpb = 1.4300 (worst) | 3.4x slower/step = catastrophic undertrain. Same as Kronecker — tiny matmuls don't saturate GPU. |
| Hybrid attn/conv (skip attention in some layers) | 1.3202-1.3241 (all worse than full attn) | Removing attention hurts more than extra conv-only depth helps. Attention is irreplaceable. |
| Ultra-thin layers (16-18L, KV2, 768h) | 1.3367-1.3199 (all worse) | Reducing MLP width + KV heads loses more per-layer quality than extra depth gains. |
| Ultra-thin layers (15-16L, KV4, 768-896h) | 1.3332-1.3363 (worse + over 16MB) | Even keeping KV4, thinner MLP doesn't beat 14L full-width. Artifact over limit. |
| LoRA Towers 20L/r64 (shared base + LoRA) | ~1.46 (much worse) | Shared weights can't match independent per-layer weights even with rank-64 residuals. Progressive gate didn't help. |

---

## C3 ConvFFN Tuning Results (all with sliding window eval, stride=64)
| Config | val_bpb | Artifact | ms/step | Notes |
|--------|---------|----------|---------|-------|
| Baseline 9L/3x (no conv) | 1.3267 | 13.1MB | 125 | Reference |
| **ConvFFN 12L/2x k=3** | **1.3218** | 14.3MB | 201 | Best at 12L |
| ConvFFN 12L/2x k=5 | 1.3229 | 14.3MB | 204 | k=3 wins |
| ConvFFN 12L/2x k=7 | 1.3244 | 14.1MB | 208 | Larger kernel = worse |
| ConvFFN 10L/3x k=3 | 1.3162 | ~15.7MB | ~165 | Wider but fewer layers, tight on budget |
| **ConvFFN 13L/2x k=3** | **1.3178** | ~14.9MB | ~217 | Safe pick (comfortably under 16MB) |
| **ConvFFN 14L/2x k=3** | **1.3152** | ~15.7MB | 237 | Best BPB, tight on 16MB |
| Hybrid 16L attn_every=2 | 1.3202 | 14.3MB | 197 | Worse than 13L full attn |
| Hybrid 18L attn_every=3 | 1.3235 | 14.7MB | 196 | Worse than 12L full attn |
| Hybrid 20L attn_every=4 | 1.3241 | 15.2MB | 202 | Extra depth doesn't help without attn |

---

## Ultra-Thin & LoRA Towers Experiments (with sliding window eval, EVAL_FRAC=0.05 or 0.2)

**Hypothesis**: Trade per-layer width for more depth, or share weights across layers with low-rank per-layer residuals.

| Config | val_bpb | Artifact | ms/step | GPUs | Notes |
|--------|---------|----------|---------|------|-------|
| Baseline 14L/KV4/2x ConvFFN | **1.3152** | **15.7MB** | 237 | 2xA30 | Current best |
| Ultra-thin 16L/KV2/768h ConvFFN | 1.3367 | 14.9MB | 475 | 2xA30 | KV2+thin MLP too aggressive |
| Ultra-thin 18L/KV2/768h ConvFFN | ~1.3199 | **16.5MB** | 536 | 2xA30 | Over 16MB, SSH dropped at 80% eval |
| 16L/KV4/768h ConvFFN | 1.3363 | **16.2MB** | 254 | 4xA30 | Over 16MB by 200KB |
| 15L/KV4/896h ConvFFN | 1.3332 | **16.4MB** | 244 | 4xA30 | Over 16MB by 400KB |
| LoRA Towers 20L/r64 ConvFFN | ~1.46 (step 1500) | ~8MB | 394 | 4xA30 | Much worse, SSH dropped |
| LoRA Towers 24L/r32 ConvFFN | *pending* | ~6MB | ~450 | 4xA30 | Running in background |

### What We Learned
1. **Reducing MLP width hurts more than extra depth helps** — 768h (1.5x) or 896h (1.75x) per layer is not enough to compensate for loss of expressivity, even with 15-18 layers vs 14.
2. **Reducing KV heads from 4→2 is also costly** — attention quality matters. Combined with thin MLP it's doubly bad.
3. **15L/896h is closest to beating baseline** (1.3332 vs 1.3152) but artifact is 400KB over 16MB. Better compression could make this viable.
4. **LoRA Towers is a clear negative result** — shared base weights + low-rank residuals (even rank-64) can't match independent per-layer weights. At 20L/r64, BPB is 1.46 vs baseline 1.32. The prior SVD analysis ("layers share zero structure") was correct even when training with sharing from scratch.
5. **Progressive gate didn't save LoRA Towers** — the 0→1 warmup (ALBERT-like early → full LoRA late) wasn't enough to overcome the fundamental expressivity limit of shared weights.
6. **DDP requires `find_unused_parameters=True` for LoRA Towers** — shared_attn's q_gain isn't used in LoRABlock forward (each block has its own), causing DDP to crash.

---

## Architecture Experiments (without sliding window)
| Architecture | val_bpb | Artifact | ms/step | Notes |
|-------------|---------|----------|---------|-------|
| Baseline 9L/3x MLP | 1.3611 | 12.5MB | 125 | Reference |
| ConvFFN 12L/2x k=3 | **1.3559** | 13.4MB | 201 | Only winner |
| Factorized FFN 14L/r192 | 1.4033 | 13.6MB | 179 | Dead end |
| Soft MoE 9L/4exp | 1.3987 | 13.0MB | 128 | Dead end |
| Butterfly 18L Monarch | 1.4300 | 10.9MB | 430 | Dead end |

---

## Compression Comparison (same 3x MLP model)
| Method | Artifact | val_bpb | Notes |
|--------|----------|---------|-------|
| int8 + zlib | 18.7MB | 1.3213 | Over limit |
| int5/6 + zstd-22 | 13.3MB | 1.3272 | Under limit |
| int5/6 + zstd + shuffle + prune 3% | **13.0MB** | **1.3275** | Current best |
| TurboQuant 3-bit | 7.9MB | 2.6098 | Broken |

---

## Key Learnings
1. **Compression is maxed at ~13MB** for this model size. More pruning doesn't help. Spend the 3MB free space on more capacity instead.
2. **torch.compile is critical** — 3x speedup means 3x more training steps in the 10-min budget. Any technique that breaks compile is dead.
3. **Sliding window eval is the biggest free lunch** — -0.034 bpb for zero cost.
4. **Quantization precision matters by layer type** — MLPs tolerate int5, attention needs int6, embeddings need fp16.
5. **Kronecker/tensor decomposition is theoretically elegant but practically bad** for this challenge — small matmuls can't saturate H100s and break torch.compile.
6. **Transformer layers do NOT share weight structure** — SVD analysis proves layers are fundamentally different (rank-358 needed to capture residuals). Any weight-sharing / factorization scheme across layers is a dead end for this architecture.
7. **Causal depthwise conv is a real improvement** — cheap local cross-token mixing (1536 params/layer) enables trading MLP width for depth. 12L/2x+conv beats 9L/3x by 0.005 bpb.
8. **More depth wins, but every layer needs attention** — hybrid (attention-free) layers are a trap. Removing attention from any layer hurts more than the extra depth helps.
9. **Kernel size 3 is optimal for depthwise conv** — k=5, k=7 both worse. Local context of 3 tokens is enough; larger kernels add noise.
10. **Step speed matters enormously** — at 2000 iterations, 200ms/step vs 125ms/step means the same wall-clock trains the same number of steps but slower per-step models see fewer tokens per second. At full scale (20K steps, 10 min budget), slower models may not finish.
11. **Ultra-thin layers don't pay off** — reducing MLP from 2x→1.5x (768h) or 1.75x (896h) to fit more layers hurts per-layer quality more than extra depth helps. 14 layers at full 2x width is the sweet spot.
12. **LoRA Towers (shared base + low-rank residuals) is a dead end** — even rank-64 residuals can't differentiate layers enough. Shared weights fundamentally limit expressivity. This confirms the SVD analysis from a different angle: training WITH sharing from scratch doesn't help either.
13. **15L at near-full width is tantalizingly close** — 15L/896h gets 1.3332 BPB but artifact is 400KB over 16MB. Better compression (not architecture) is the path to more depth.

---

## Test Environment
- GPU: 1-4x A30 (DSMLP n11, no MIG), PyTorch 2.4 with torch.compile
- Scale: 2000 iterations, batch=131072, seq_len=1024, 40 data shards (4B tokens)
- This is ~25-30% of full competition compute (8xH100, 20K steps, 8B tokens)

## Leaderboard Reference
- Naive baseline (8xH100): 1.2244
- Current merged SOTA: ~1.1194
- Best pending PR: ~1.1063
- Our best: **1.3152** (14L ConvFFN, small scale, tight on 16MB)
- Our safe best: **1.3178** (13L ConvFFN, small scale, under 16MB)
