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

## Quantitative Results (at ~28 steps)

| Experiment | Steps | R>0 Rate | Grad Rate | Reward Trend (Q1→Q4) | Learning? |
|-----------|-------|----------|-----------|---------------------|-----------|
| **titan_simple** | 28 | **86%** | **86%** | 0.65 → 0.65 → 0.38 → 0.38 | Declining after initial signal |
| regr_simple | 29 | 52% | 55% | 0.01 → 0.01 → 0.03 → 0.01 | Flat |
| bos_simple | 24 | 25% | 29% | 0.09 → 0.08 → 0.01 → 0.01 | Declining |
| **titan_tiered** | 29 | **86%** | 21% | 0.68 → 0.64 → 0.97 → 0.86 | **Increasing** |
| bos_tiered | 29 | 38% | 24% | 0.43 → 0.15 → 0.00 → 0.55 | Volatile |
| regr_tiered | 28 | 50% | 32% | 0.06 → 0.11 → 0.29 → 0.06 | Flat |
| titan_voi | 24 | 21% | 42% | 0.02 → 0.02 → 0.00 → 0.07 | Slight increase |
| bos_voi | 25 | 4% | 8% | 0.03 → 0.00 → 0.00 → 0.00 | No signal |
| regr_voi | 25 | 12% | 32% | 0.02 → 0.00 → 0.00 → 0.03 | Marginal |

### Key Finding: Simple reward has highest gradient rate, but reward is declining

The simple reward produces the most gradient updates (86% for Titan) because continuous rewards create variance across K=4 proposals. However, the reward is declining in Q3-Q4, suggesting:

1. The model is being pulled away from SFT-quality proposals by the gradient signal
2. The reward is based on keyword matching to historical trees, which is noisy
3. The KL penalty (0.1) may not be strong enough to prevent drift

### Tiered reward has lower gradient rate but more stable learning

Tiered shows increasing reward on Titan despite fewer gradient steps. The discrete rewards (0, 0.2, 1.0) only produce gradient when proposals land in different tiers, which is a stronger but rarer signal.

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
