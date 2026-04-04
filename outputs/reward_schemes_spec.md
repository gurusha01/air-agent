# RL Reward Schemes — Canonical Spec

All schemes evaluate a single episode's executor score `s` against a fixed
baseline `b` (loaded from the task YAML, e.g., `b = 0.8478` for FashionMNIST).
`s = None` iff the executor faulted and no numeric score was produced.

All schemes assume **higher is better**; the env flips inequalities internally
for lower-is-better tasks.

Stateful schemes (v8, v9) use **end-of-step snapshot** semantics:
all `batch_size` rollouts within a training step are scored against the
state as it was at the start of that step. After all rollouts in the step
have been scored, the state is committed using the step's scores.

---

## v6 — binary (stateless)

Already trained (43 steps, `rl_v6_fmnist`, timed out at 8h).

```
r(s) = -0.5   if s is None
     =  0.0   if s <= b
     = +1.0   if s >  b
```

---

## v7 — fixed tier (stateless)

Constant `τ = 0.88` hard-coded in `V7_TAU` in `air/ttt/mlgym_env_v2.py`.

```
r(s) = -0.5   if s is None
     =  0.0   if s <  b
     = +0.2   if b <= s <= τ
     = +1.0   if s >  τ
```

---

## v8 — global monotone best_ever (stateful, persistent)

Maintain `best_ever` initialized to `b`, monotonically non-decreasing.

Within a training step, all rollouts see the `best_ever` snapshot from the
previous step's commit. At commit time:

```
best_ever ← max(best_ever, max of this step's valid scores)
```

Reward:

```
r(s) = -0.5   if s is None
     =  0.0   if s <= b
     = +0.2   if b <  s <  best_ever
     = +1.0   if s >= best_ever      (ties at current best_ever get +1)
```

State persisted to `/scratch/jarnav/rollout_logs/reward_state_v8_{task}.json`.

---

## v9 — percentile rolling window (stateful, persistent)

Maintain a FIFO deque `window` of the last `N = 64` **valid raw scores**
(including scores below baseline; nothing is clipped). Threshold:

```
p = b                          if |window| < warmup   (warmup = 8)
  = percentile(window, q)      otherwise              (q = 70)
```

All rollouts in a training step use the `p` snapshot from step start.
At commit time, the step's valid scores are appended to `window` (FIFO).

Reward:

```
r(s) = -0.5   if s is None
     =  0.0   if s <= b
     = +0.2   if b <  s <= p
     = +1.0   if s >  p
```

State persisted to `/scratch/jarnav/rollout_logs/reward_state_v9_{task}.json`.

---

## Tie conventions (summary)

| Scheme | `s = b`   | `s = τ / best_ever / p` |
|--------|-----------|-------------------------|
| v6     | 0         | — |
| v7     | +0.2      | `s = τ` → +0.2 (top tier requires strict `>`) |
| v8     | 0         | `s = best_ever` → +1 (`>=`) |
| v9     | 0         | `s = p` → +0.2 (top tier requires strict `>`) |

---

## Run log

| Scheme | Config | Job | Status |
|---|---|---|---|
| v6 | `prime_rl_v6_fmnist.toml` | 4611953 | TIMEOUT at 8h, 43 steps done |
| v7 | `prime_rl_v7_fmnist.toml` | TBD | pending |
| v8 | `prime_rl_v8_fmnist.toml` | TBD | pending |
| v9 | `prime_rl_v9_fmnist.toml` | TBD | pending |
