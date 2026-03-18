# Scientist Finetuning Plan (v2)

Training method: **PPO or GRPO with online/partially-offline rollouts, LoRA adapters.**
No DPO, no offline RL.


---

## Experiment 1: Exploration / Exploitation as Information Gain

### The Problem with Naive Uncertainty Measures

Score variance across nodes is **not** uncertainty about a direction. A direction can have:
- Low score variance but high conceptual uncertainty (tested one config of "ensembles" — what about other ensemble types?)
- High score variance but low conceptual uncertainty (tried many configs, we understand the landscape)

Uncertainty must be about **the idea itself** — specifically, how much of the idea's
hypothesis space has been tested, and whether the experiments so far can distinguish
between competing explanations.

### When Do Great Scientists Commit?

Not on a schedule. The trigger is **convergence of beliefs**, not exhaustion of budget.

A scientist explores when:
- Outcomes are surprising (their world model is wrong → more information to gain)
- Key hypotheses remain untested (the hypothesis space is underexplored)
- No single direction clearly dominates (ambiguity about what works)

A scientist exploits when:
- They understand *why* the best direction works (causal model, not just high score)
- Further experiments in the best direction have *predictable* outcomes (low information gain)
- The gap between best and second-best is large and stable

From cognitive science:
- **Directed exploration** (Wilson et al., 2014): humans explore based on *uncertainty*
  about an option, not just value — they seek information where their model is weakest.
- **Satisficing** (Simon, 1956): humans don't optimize forever; they commit when a direction
  is "good enough" relative to their aspiration level (which depends on the task).
- Information-theoretic accounts (Gottlieb et al., 2013): curiosity is the drive to reduce
  uncertainty; it naturally decays as the world model converges.

### Reward Design (v2)

```
R(step t) = α(state_t) · R_explore(direction_t) + (1 - α(state_t)) · R_exploit(step t)
```

#### α(state_t) — state-dependent, not time-dependent

```
α(state_t) = σ(w · features(state_t))

features:
  - mean_residual_surprise    ← high → explore more (world model is bad)
  - budget_fraction_left      ← low → exploit more (time pressure)
  - best_improvement_so_far   ← high → exploit (found something worth deepening)
  - fraction_successful_nodes ← low → explore differently (env problems, not idea problems)
  - hypothesis_coverage       ← low → explore (untested hypotheses remain)

σ = sigmoid, w ∈ R^5 learned jointly with the policy.
```

The key: α is learned, not hand-tuned. The model learns from experience when to explore
and when to exploit, using features that capture the relevant aspects of the tree state.
Budget fraction is just ONE input — the model can learn to override time pressure
when uncertainty is still high.


#### R_explore — information value of the direction, evaluated BEFORE seeing results

The experiment's value is about its *design*, not its outcome.

**Structured scientist output** (new fields added to the prompt):
```
HYPOTHESES_TESTED: [which beliefs about the task does this experiment address?]
EXPECTED_IF_TRUE:  [predicted score range if the hypothesis is correct]
EXPECTED_IF_FALSE: [predicted score range if the hypothesis is wrong]
```

Then:
```
R_explore = discrimination × novelty

discrimination = |E[score | H_true] - E[score | H_false]|
  → high when the experiment can distinguish between world models
  → an experiment where both outcomes give ~0.90 teaches nothing
  → computed from the scientist's own predictions

novelty = 1 - max_cosine_sim(hypothesis_embed, all_prior_hypothesis_embeds)
  → prevents restating the same hypothesis with different words
  → can use a small embedding model or structured category tags
```

**Why this works**: An experiment with high discrimination is a *good experimental design* —
it will be informative regardless of outcome. An experiment with high novelty tests a
region of hypothesis space we haven't explored. Together, they measure "how much will I
learn from this?" which is exactly what exploration should maximize.

**Post-hoc verification**: After observing the result, check whether the scientist's
predicted ranges were sensible. If the scientist says "expected 0.85–0.90 if true,
0.70–0.75 if false" and the result is 0.88, then this confirms the hypothesis. If the
result is 0.50 (outside both ranges), the scientist's hypotheses were wrong, but at
least the experiment was designed to be discriminative — still worth partial credit.


#### R_exploit — genuine improvement with execution quality gate

```
R_exploit(step t) = improvement × execution_quality × depth_bonus

improvement = (score_t - best_score_before_t) / max(|best_score_before_t - baseline|, ε)
  → normalized so +0.01 near ceiling is worth more than +0.01 near baseline
  → 0 if no improvement (no penalty — exploitation that confirms plateau is still information)

execution_quality = {
   1.0   if execution_status == "success"
   0.0   if execution_status == "training_failed"  ← zero reward for fallback scores
  -0.1   if execution_status in ("no_submission", "no_validate")  ← budget wasted
}

depth_bonus = 1 + 0.1 × depth
  → small bonus for deepening a promising branch vs restarting from root
  → directly counteracts FM1 (root-restart death spiral)
```

The execution_quality gate is critical: it completely nullifies reward for fallback
baseline scores. The scientist can no longer get "exploit signal" from a crashed
training run that happens to score 2.38 via the pre-existing submission file.

**Note on improvement normalization**: we divide by `|best - baseline|` so that:
- On a task where best=0.93 and baseline=0.87, a +0.01 gain = 0.01/0.06 = 0.17 (big deal)
- On a task where best=0.55 and baseline=0.50, a +0.01 gain = 0.01/0.05 = 0.20 (comparable)
This makes the reward comparable across tasks, important for multi-task training.


---

## Experiment 2: World Model as Bayesian Belief Maintenance

### Framing

The scientist maintains a **posterior distribution over hypotheses about the task**.
Memory is the natural-language representation of this posterior.

```
Before any experiments:  P(H1), P(H2), ..., P(Hk)   — prior (from task description)
After experiment i:       P(Hi | data_1..i)           — posterior (stored as memory)
```

A "correct" memory update is one that performs a proper Bayesian update:
- Strengthens hypotheses supported by new evidence
- Weakens hypotheses contradicted by new evidence
- Preserves hypotheses not tested by this experiment
- Does NOT overwrite good beliefs with noise from one data point

### Reward: Surprise Absorbed, Not Just Surprise

(Following user's insight)

The reward should measure: **after updating your beliefs, how much of the surprising
observation can you now explain?**

```
R_memory(step t) = surprise_absorbed(t)
                 = surprise_before_update(t) - surprise_after_update(t)
```

**surprise_before_update(t)**: How surprising was the outcome of step t, given the
scientist's beliefs (memory) BEFORE step t?

```
surprise_before(t) = prediction_error(memory_{t-1}, outcome_t)
```

**surprise_after_update(t)**: After the scientist updates memory at step t, does the
new memory better account for all observations so far? Measured prospectively:

```
surprise_after(t) = prediction_error(memory_t, outcome_{t+1})
  → at step t+1, how surprised is the scientist with its updated beliefs?
```

So:
```
R_memory(t) = |pred(memory_{t-1}) - outcome_t| - |pred(memory_t) - outcome_{t+1}|
```

**Interpretation**: If you saw something surprising and your memory update now explains it,
your future predictions improve → positive reward. If your memory update is wrong,
irrelevant, or fails to incorporate the new information, future surprise stays high
→ zero or negative reward.

This incentivizes the scientist to:
1. Notice surprising outcomes (requires attention to execution results)
2. Update beliefs to explain them (requires correct causal reasoning)
3. Use updated beliefs to make better predictions (requires coherent world model)

### Making Predictions Explicit

To compute surprise, we need the scientist's predictions. Add to the structured output:

```
PREDICTED_SCORE_RANGE: [low, high]
  → scientist's predicted score range for this experiment
  → based on current memory/beliefs
```

Then:
```
surprise(prediction, outcome) = max(0, outcome - high) + max(0, low - outcome)
  → 0 if outcome is within predicted range (not surprised)
  → positive if outcome falls outside (surprised)
```

This directly measures calibration of the world model. A well-calibrated scientist
predicts wide ranges when uncertain and narrow ranges when confident — and is right
about both.

### The Residual Surprise Idea (Key Contribution)

Standard approach: penalize surprise → R = -surprise(t).
Problem: this incentivizes the model to predict very wide ranges (never surprised, but useless).

User's insight: reward **surprise absorbed** = how much surprise is REDUCED by the update.

```
R_memory(t) = surprise(memory_{t-1}, outcome_t) - surprise(memory_t, outcome_{t+1})
```

This avoids the "predict everything" failure mode:
- Wide predictions at step t → low surprise_before(t) → low R_memory (nothing to absorb)
- Narrow correct predictions → low surprise_before AND low surprise_after → small R_memory
  (you were already calibrated, not much to gain)
- Surprised → then updated beliefs → less surprised next time → LARGE R_memory (learned something!)

The reward is maximized when: (1) the scientist encounters something unexpected, AND
(2) it correctly updates its world model to account for it.

### What a Good World Model Looks Like (Bayesian Lens)

```
Step 0: "I hypothesize that feature engineering helps more than model choice for this
         tabular dataset."  [P(features_matter) = 0.7, P(model_matters) = 0.3]

Step 1: Tried CatBoost, got 0.88. Similar to LightGBM's 0.87.
         → Evidence: changing model doesn't help much.
         → Update: P(model_matters) ↓ to 0.1, P(features_matter) ↑ to 0.9
         → Memory: "Switching models gives <0.01 improvement. Feature engineering
           is the likely bottleneck."

Step 2: Tried feature engineering (polynomial features), got 0.93.
         → Evidence: confirms features hypothesis.
         → Update: P(features_matter) ↑ to 0.95
         → Memory: "Polynomial features on age/fare gave +0.05. Model choice confirmed
           secondary. Next: try interaction features or target encoding."

Step 3: Tried interaction features, got 0.92 (slightly worse than polynomial).
         → Not surprising given beliefs (polynomial was specific, interactions are general).
         → Update: small adjustment. Polynomial > interactions.
         → Memory: "Polynomial features (0.93) > interaction features (0.92).
           Specific nonlinear transforms beat generic interactions."
```

At each step, the posterior narrows and the predictions get more accurate. THIS is what
we want the LoRA to learn to do.


---

## Unified Reward Function

```
R(step t) = α(state_t) · R_explore + (1 - α(state_t)) · R_exploit + β · R_memory

where β is a fixed hyperparameter (e.g., 0.3) or also state-dependent.
```

All three components are computable from the trajectory using the structured outputs
(HYPOTHESES_TESTED, EXPECTED_IF_TRUE/FALSE, PREDICTED_SCORE_RANGE) and the env feedback
(execution_status, error_type) we just added to the codebase.


---

## Training Setup: GRPO with Online Rollouts

### Why GRPO over PPO

- Value function is hard to train when the state is a long text prompt (tree view + memory)
- GRPO estimates advantages from group comparison: no value function needed
- Successfully used for LLM finetuning at scale (DeepSeek-R1)
- Works well with LoRA

### GRPO Procedure

For each training iteration:
1. Sample a task T and initial tree state S
2. Generate K=4 rollouts from the same starting state using the current LoRA policy
3. Execute all K in parallel MLGym containers (each rollout = full search: 5 or 15 steps)
4. Compute per-step and cumulative reward for each rollout
5. Rank the K rollouts by total episode reward
6. Normalize advantages: A_i = (R_i - mean(R)) / std(R) within the group
7. Update LoRA weights: ∇ = Σ_i A_i · ∇log π(actions_i | states_i)

### LoRA Configuration

- Base model: Qwen3-4B-Instruct-2507 (or 8B if budget allows)
- LoRA rank: 16–32
- Target modules: q_proj, v_proj, k_proj, o_proj
- ~10–30M trainable params
- KL penalty from base model to prevent mode collapse

### Partially Offline Warmup

Stage 0 (warm start):
- Collect the best existing trajectories (o3o3 runs with genuine improvements)
- SFT the LoRA on these trajectories for 2–3 epochs
- This teaches the model the output format and basic decision patterns

Stage 1 (online GRPO):
- K=4 rollouts per (task, state) pair
- 5 non-RL tasks × 2 budget sizes = 10 configurations
- ~4 groups per iteration = 40 rollouts = ~20 GPU-hours per iteration
- Run 10–20 iterations

### Rollout Cost Estimate

Per rollout (n=5):
  - 5 scientist calls @ ~1s each (Qwen via vLLM) = 5s
  - 5 executor episodes @ ~3 min each (container + LLM) = 15 min
  - Total: ~15 min per rollout

Per iteration (K=4 rollouts × 10 configs):
  - 40 rollouts × 15 min = 10 GPU-hours (with 4-way parallelism: 2.5 wall hours)

For 15 iterations: ~150 GPU-hours total (feasible on the HPC cluster).


---

## Structured Scientist Output (Updated Prompt)

The scientist currently outputs: REASONING, STRATEGIES, CHOSEN, DIRECTION, MODE, MEMORY.

New required fields for the reward to be computable:

```
HYPOTHESES_TESTED: [which hypotheses does this experiment address?]
EXPECTED_IF_TRUE:  [score range if the primary hypothesis is correct]
EXPECTED_IF_FALSE: [score range if the primary hypothesis is wrong]
PREDICTED_SCORE_RANGE: [low, high] — overall prediction for this node
MODE: explore | exploit
MEMORY: [updated belief — what did I learn, what do I now believe?]
```

These are needed both for R_explore (discrimination value) and R_memory (surprise computation).
They also make the scientist's reasoning more transparent and debuggable.


---

## Priority & Sequencing

| Step | What | Compute | Timeline |
|------|------|---------|----------|
| 0 | Add structured output fields to scientist prompt | 0 | Now |
| 1 | Run new experiments with env feedback (already submitted) | 130 jobs | ~1 day |
| 2 | Mine existing + new trajectories for warm-start data | CPU only | 1 day |
| 3 | SFT warm start on best trajectories | 1 GPU, few hours | 1 day |
| 4 | Implement GRPO training loop with MLGym env | Engineering | 2–3 days |
| 5 | Online GRPO training (15 iterations) | ~150 GPU-hours | 3–5 days |
| 6 | Evaluate: LoRA scientist vs base scientist vs o3 scientist | 30 runs | 1 day |
