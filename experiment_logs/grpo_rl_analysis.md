# GRPO/RL Training Analysis Report

**Date**: 2026-03-24
**Status**: 9 experiments running on gpubase_interac (L40S GPUs)

---

## Experimental Setup

Three reward functions × three tasks = 9 experiments:

| Reward | Description | Gradient Signal |
|--------|-------------|-----------------|
| **Simple** | `(sim_score - baseline) / \|baseline\|`, clipped at 0 | Continuous → high variance across K=4 proposals → frequent gradients |
| **Tiered** | 0 (no gain), 0.2 (small gain), 1.0 (big jump, >5%) | Discrete → often all 4 proposals get same tier → sparse gradients |
| **VoI** | R_explore + R_exploit + R_memory | Complex → requires valid parse + prediction → very sparse reward |

All experiments: Qwen3-4B + LoRA (r=16), per-task SFT v3 epoch3 checkpoints, K=4 proposals, node_budget=4, 200 steps.

---

## Quantitative Results (at ~50 steps, updated 21:45 MDT)

| Experiment | Steps | R>0 Rate | Grad Rate | 5-Bucket Trend | Learning? |
|-----------|-------|----------|-----------|----------------|-----------|
| **titan_simple** | 52 | **83%** | **83%** | 0.67 → 0.50 → 0.30 → 0.69 → 0.49 | Degrading (overfit to keywords) |
| regr_simple | 54 | 65% | 67% | 0.01 → 0.02 → 0.01 → 0.02 → 0.01 | Stable (low signal) |
| bos_simple | 49 | 37% | 53% | 0.12 → 0.01 → 0.04 → 0.09 → 0.01 | Degrading |
| titan_tiered | 54 | 83% | 24% | 0.69 → 0.82 → 0.83 → 0.80 → 0.68 | Stable (high reward) |
| bos_tiered | 55 | 45% | 45% | 0.30 → 0.07 → 0.56 → 0.18 → 0.31 | Volatile |
| **regr_tiered** | 53 | 62% | 32% | **0.10 → 0.17 → 0.13 → 0.19 → 0.16** | **LEARNING ↑** |
| titan_voi | 50 | 12% | 28% | 0.02 → 0.03 → 0.01 → 0.00 → 0.01 | Degrading |
| bos_voi | 51 | 4% | 10% | 0.02 → 0.00 → 0.00 → 0.00 → 0.01 | No signal |
| regr_voi | 50 | 16% | 44% | 0.01 → 0.01 → 0.04 → 0.01 → 0.01 | Stable (low signal) |

### Key Finding 1: Simple reward has highest gradient rate, but rewards degrade

The simple reward produces the most gradient updates (83% for Titan) because continuous rewards create variance across K=4 proposals. However, rewards decline over training — the model learns keyword patterns that match historical trees but this doesn't generalize. The policy drifts from SFT quality.

### Key Finding 2: Tiered reward is more stable but gradient-sparse

Tiered shows high reward rates but low gradient rates (24% for Titan). All 4 proposals often land in the same tier → zero advantage → no gradient. When gradients do occur, they're stronger signals.

### Key Finding 3: Only regr_tiered shows clear learning

Regression with tiered reward is the only experiment where reward consistently increases (0.10 → 0.19). This may be because regression has the most structured score landscape — small improvements are reliably in the 0.2 tier, large improvements in the 1.0 tier.

---

## Qualitative Analysis

### What high-reward proposals look like

```
type: explore
hypothesis: Missing Age values are not randomly distributed and are correlated
with other features like Pclass and Embarked, leading to a bias in survival
prediction if not properly handled
experiment: Impute missing Age values using median grouped by Pclass and Sex...
```

These proposals mention specific features (Age, Pclass, Embarked) that overlap with keywords in historical tree nodes → high simulated score.

### What zero-reward proposals look like

```
## Constraints
- You cannot run more than 4 experiments total.
- You must include a thought_update in every experiment.
- Do not include any explanations beyond the specified format.
```

The model generates meta-instructions instead of actual experiment proposals. These have no feature keywords → no historical match → sim_score=None → reward=0.

### What's happening over training

**Early steps**: Model outputs a mix of actual proposals and meta-text. Proposals that mention domain-specific features (Age, Pclass, GrLivArea) get rewarded via keyword matching.

**Later steps**: Model is converging toward short meta-instructions ("Only one experiment per turn", "Do not output anything else"). This is likely because:
1. The 128-token limit truncates proposals → only the beginning matters
2. Meta-text occasionally gets keyword matches by chance → noisy reward
3. The gradient is pushing toward shorter outputs (less KL cost)

---

## Technical Issues Encountered

### 1. CUDA OOM on L40S (44GB VRAM)
- **Cause**: Two Qwen3-4B models (bf16) = 16GB, plus activations for K=4 proposals
- **Fix**: Added gradient checkpointing + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + ref model in float16
- **Result**: Most jobs now survive on L40S, though some still OOM on longer prompts

### 2. Full finetuning OOM
- **Cause**: AdamW needs 8 bytes/param × 4B params = 32GB for optimizer states alone
- **Impact**: Full FT requires A100 (80GB). All full-FT jobs OOM'd on L40S.
- **Decision**: Kept LoRA for all experiments

### 3. CPU-only training (original bug)
- **Cause**: Models never moved to GPU (no `.to(device)` call)
- **Fix**: Added `device = torch.device("cuda")` and `.to(device)` for both models
- **Impact**: 5.5x speedup (470s → 85s per step)

### 4. Cluster congestion
- **bygpu partition**: 2,577 pending jobs
- **Solution**: Used `gpubase_interac` partition (0 pending, 10 idle L40S nodes)

---

## Score Simulator Issues

The `simulate_score` function uses keyword overlap between proposal text and historical tree node strategies. This creates several problems:

1. **Bogus scores**: Titanic `best=2.39` (accuracy can't exceed 1.0). The simulator returns historical scores with noise, which can exceed valid ranges.
2. **No real execution**: Proposals aren't actually run in MLGym containers. The "simulated score" is a keyword-matching heuristic.
3. **Same score for all K proposals**: When all 4 proposals use similar domain keywords, they match the same historical node → same score → zero advantage → no gradient.

---

## Conclusions and Next Steps

### What works
1. **Simple reward with continuous signal** produces the highest gradient rate (86% for Titan)
2. **GPU training** is 5.5x faster than CPU (85s vs 470s per step)
3. The model **does produce non-zero gradients** and reward trends upward in some configurations

### What doesn't work
1. **VoI reward** is too complex — only 4-21% non-zero reward. The parse → predict → compare pipeline has too many failure modes.
2. **Tiered reward** has high reward rate but low gradient rate because all K=4 proposals often land in the same tier
3. **BoS (game theory)** doesn't learn well with any reward — the keyword-matching simulator doesn't capture game theory reasoning quality

### What to try next
1. **Better score simulator**: Use the actual MLGym container to execute proposals (expensive but realistic)
2. **Increase K from 4 to 8**: More proposals → more likely to get variance → more gradient
3. **Reduce KL coeff**: Currently 0.1, might be too high and constraining the policy too much
4. **Increase max_new_tokens**: 128 tokens often truncates the actual experiment proposal
5. **Format-aware reward**: Add a small bonus (0.1) for proposals that successfully parse as valid scientist output, regardless of score
