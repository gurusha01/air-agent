# Best Trees: o3 on Mountain Car (reward)

## AIRA (best variant)

**Experiment:** `o3_aira_evo.g` | **Best Score:** 49.7180 | **Nodes:** 9

*Other variants:* `o3_aira_greedy.g` (49.7180)

```
`root` [33.7938]
  ↳ Baseline: PPO with default config (20M steps, 256 envs, 4 layers of 256 units, lr=6e-4, gamma=0.99, gae_lambda=0.99, entropy_coeff=0.1).
├── `root_0` [FAIL]
│     ↳ Attempted to add action clipping (jnp.clip to [-1,1]) in select_action and batch_step, but sed inserted clip into wrong function (batch_reset) causing UnboundLocalError. Then tried velocity-based heuristic bias in actor network but sed produced SyntaxError.
├── `root_1` [FAIL]
│     ↳ Read source code extensively. Attempted to fix log_prob shape mismatch and reduce training to 1M steps. Got -60.0 avg reward. Reverted and tried 4M steps but reintroduced log_prob bug. Final validate failed because checkpoints were deleted before training completed.
├── `root_2` [49.7180] **<-- BEST**
│     ↳ Fixed critical log_prob action shape issue: changed pi.log_prob(action[...,-1]) to pi.log_prob(action) and fixed action dimension expansion. Kept default config. Training produced mixed results (80, -0.04, 76, -0.12, 92 across 5 seeds). Shape fix was the key breakthrough.
│   ├── `root_2_0` [FAIL]
│   │     ↳ Container state broken -- src/ directory missing from workspace. Agent spent all 20 steps searching for files and outputting natural-language suggestions as bash commands.
│   ├── `root_2_1` [FAIL]
│   │     ↳ Same broken environment -- src/ directory missing. Repeatedly tried to find source files, issued natural-language suggestions as bash commands. Never wrote code.
│   └── `root_2_2` [FAIL]
│         ↳ Same broken environment. Searched exhaustively for source files. Never modified any code.
├── `root_3` [FAIL]
│     ↳ Fixed log_prob shape (same fix as root_2). Tried multiple configs: (1) 400K steps, 128 envs, 3 layers of 128 -> -3.2 avg; (2) 3M steps, 256 envs, 4 layers of 256 -> -0.47 avg; (3) reverted to 500K steps but also removed action.squeeze() in evaluation which broke things. Ran out of steps.
└── `root_4` [FAIL]
      ↳ Read source files extensively (5 reads of policy.py). Lost workspace midway (src/ files disappeared). Never produced checkpoints.
```

## UCB + VS

**Experiment:** `o3_ucb.g` | **Best Score:** 33.7938 | **Nodes:** 9

```
`root` [33.7938] **<-- BEST**
  ↳ Baseline (no model execution)
├── `root_0` [FAIL]
│     ↳ Curriculum-learning with penalty annealing: start PPO training on an easier MountainCarContinuous va
├── `root_1` [FAIL]
│     ↳ Replace the default MLP actor-critic with a linear policy/value function built on top of a fixed Rad
├── `root_2` [FAIL]
│     ↳ Strategy: introduce State-Dependent Exploration (SDE) in PPO (“PPO-SDE”) plus observation/action nor
├── `root_3` [FAIL]
│     ↳ Variation – Add Layer Normalization to the PPO policy/value networks and use a linear learning-rate
├── `root_4` [FAIL]
│     ↳ Variation: Introduce state-value normalization + slightly more aggressive entropy regularization.

1
├── `root_5` [FAIL]
│     ↳ Variation: “Large‐capacity dual-head network with aggressive normalization”

1. Network architecture
├── `root_6` [FAIL]
│     ↳ Variation: Use an “aggressive–to–conservative” annealing schedule for both the policy-clip range and
└── `root_7` [FAIL]
      ↳ Entropy-annealing PPO: keep all current PPO hyper-parameters but add an explicit entropy-bonus sched
```

## Open-Ended Exploration

**Experiment:** `o3_oe.g` | **Best Score:** 33.7938 | **Nodes:** 9

```
`root` [33.7938] **<-- BEST**
  ↳ Baseline (no model execution)
├── `root_0` [-0.0746]
│     ↳ Model-Based Rollout Augmented PPO (MBR-PPO):  alongside the standard PPO policy/value networks, lear
│   ├── `root_0_0` [FAIL]
│   │     ↳ Potential-Based Reward Shaping PPO (PBRS-PPO): keep vanilla PPO but reshape the environment reward w
│   ├── `root_0_1` [FAIL]
│   │     ↳ Adopt an entropy-maximizing critic-based method instead of PPO: train a Soft Actor–Critic (SAC) agen
│   ├── `root_0_2` [FAIL]
│   │     ↳ Tile-Coded Natural Actor–Critic (TC-NAC)  
1. Replace the NN policy/value with separate linear funct
│   ├── `root_0_3` [FAIL]
│   │     ↳ Potential-based reward shaping PPO (Model-Free):  keep the original PPO pipeline but replace the spa
│   └── `root_0_4` [FAIL]
│         ↳ Distributional Critic PPO (QR-PPO): keep the standard PPO actor network but replace the scalar value
├── `root_1` [FAIL]
│     ↳ Apply a curriculum-learning variant of PPO: begin training on an easier version of MountainCarContin
└── `root_2` [FAIL]
      ↳ Fourier-Basis Feature Expansion for PPO:  
Instead of feeding the raw 2-dim (position, velocity) sta
```
