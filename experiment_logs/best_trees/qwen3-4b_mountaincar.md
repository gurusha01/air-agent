# Best Trees: Qwen3-4B on Mountain Car (reward)

## AIRA (best variant)

**Experiment:** `aira_mcts.g` | **Best Score:** 56.0108 | **Nodes:** 9

*Other variants:* `aira_evo.g` (51.3875), `aira_greedy.g` (51.3875)

```
`root` [33.7938]
  ↳ Baseline: PPO with default config (256 hidden units, 2 layers, lr=3e-4, gamma=0.99, gae_lambda=0.99).
├── `root_0` [56.0108] **<-- BEST**
│     ↳ Rewrote config.yaml: 512 hidden units, 4 layers, lr=0.001 (constant), 256 parallel envs, n_steps=256, 8 PPO epochs, gae_lambda=0.99, entropy_coeff=0.1. Wider/deeper network + higher entropy helps exploration. 3/5 seeds solve (reward ~93), 2/5 fail near 0.
│   └── `root_0_0` [56.0108]
│         ↳ Attempted to change hidden units from 256 to 512, but parent already set them to 512, so no actual change occurred. Retrained with identical config.
│       └── `root_0_0_0` [FAIL]
│             ↳ Tried to scale to 6 layers (from parent's 4) while keeping 512 units. Many steps wasted on broken python -c commands with empty strings. Eventually modified config but training timed out at 1800s due to larger 6-layer network.
├── `root_1` [49.7180]
│     ↳ Read source code, wasted steps on broken python -c commands. Attempted config changes but the read-only python -c command didn't write back. Ran training with unchanged baseline config.
│   └── `root_1_0` [FAIL]
│         ↳ Tried to increase layers from 4 to 6 via python string replace, but replace call emptied the YAML file (wrote None). Training crashed with NoneType error on load_config.
└── `root_2` [49.7180]
      ↳ Read source files, failed with multiple broken python -c commands, gave up on config changes and retrained with unmodified baseline config. Same result as root_1 (baseline variance).
    └── `root_2_0` [32.9641]
          ↳ Successfully changed hidden layers from 2 to 6 via yaml.safe_load/yaml.dump. Deeper 6-layer network (256 units) hurt performance -- only 2/5 seeds solve, average drops below baseline.
        └── `root_2_0_0` [32.9641]
              ↳ Read source files, failed with broken commands. Retrained with inherited 6-layer config from parent. Identical results since no config was changed.
```

## UCB + Value Synthesis

**Experiment:** `ucb_c1.g` | **Best Score:** 68.8956 | **Nodes:** 9

*Other variants:* `ucb_c2.g` (65.0326), `ucb_c1.4.g` (49.7180), `ucb_c1.p` (49.7180), `ucb_c1.4.p` (33.7938), `ucb_c2.p` (33.7938)

```
`root` [33.7938]
  ↳ Baseline (no model execution)
├── `root_0` [68.8956] **<-- BEST**
│     ↳ Introduce a hierarchical reward shaping scheme where the agent receives intermediate rewards based o
│   └── `root_0_0` [68.8956]
│         ↳ The reward shaping scheme is modified to introduce a state-dependent scaling factor for intermediate
│       └── `root_0_0_0` [68.8956]
│             ↳ Introduce a hierarchical reward function that separates the task into two distinct phases: valley es
│           └── `root_0_0_0_0` [68.8956]
│                 ↳ Introduce a physics-informed neural network (PINN) architecture that explicitly models the underlyin
│               └── `root_0_0_0_0_0` [68.8956]
│                     ↳ Introduce a trajectory-based memory network using a sequence-to-sequence architecture with a transfo
│                   └── `root_0_0_0_0_0_0` [68.8956]
│                         ↳ Replace the trajectory-based memory network with a sparse temporal coding mechanism where only signi
│                       └── `root_0_0_0_0_0_0_0` [68.8956]
│                             ↳ Introduce a hybrid latent space representation using variational autoencoders (VAEs) to compress the
└── `root_1` [-32358906.00]
      ↳ Use a Twin Delayed DDPG (TD3) agent with noise injection (Ornstein-Uhlenbeck) in the action space an
```

## Open-Ended Exploration

**Experiment:** `oe_t0.5_k2.p` | **Best Score:** 51.3875 | **Nodes:** 9

*Other variants:* `oe_t0.5_k2.g` (51.3677), `oe_t0.3_k2.p` (49.7180), `oe_t0.5_k3.p` (49.7180), `oe_t1.0_k2.p` (49.7180), `oe_t1.0_k2.g` (33.7938)

```
`root` [33.7938]
  ↳ Baseline (no model execution)
├── `root_0` [49.7180]
│     ↳ Replace the PPO agent with a Twin Delayed DDPG (TD3) actor-critic architecture, leveraging continuou
│   └── `root_0_0` [49.7180]
│         ↳ Introduce a deep deterministic policy gradient (DDPG) variant with a hierarchical action space decom
│       └── `root_0_0_0` [49.7180]
│             ↳ Replace the DDPG architecture with a Twin Delayed DDPG (TD3) variant and introduce a learned positio
└── `root_1` [51.3875] **<-- BEST**
      ↳ Use a hierarchical reinforcement learning framework where the high-level policy selects driving phas
    ├── `root_1_0` [34.9719]
    │     ↳ Replace the hierarchical policy architecture with a single, densely connected actor-critic network u
    ├── `root_1_1` [-68.4830]
    │     ↳ Replace the hierarchical policy with a single, learned policy using a Transformer-based architecture
    ├── `root_1_2` [-68.4830]
    │     ↳ Replace the hierarchical policy structure with a single, continuous policy using a Transformer-based
    └── `root_1_3` [34.9719]
          ↳ Replace the hierarchical policy architecture with a single, fully connected feedforward neural netwo
```
