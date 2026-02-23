# Best Trees: Qwen3-4B on Battle of Sexes (score)

## AIRA (best variant)

**Experiment:** `aira_mcts.g` | **Best Score:** 1.4439 | **Nodes:** 13

*Other variants:* `aira_evo.g` (1.4410), `aira_greedy.g` (1.2900)

```
`root` [1.0227]
  ↳ Baseline: runs the provided starter code.
├── `root_0` [1.0592]
│     ↳ Mostly-sticky strategy: repeats last move with 90% probability, flips with 10%. Starts with action 0. Idea is to stay predictable for the 80%-copy opponent while injecting small noise.
├── `root_1` [0.7430]
│     ↳ Round-parity alternating strategy: switches to 1 if opponent copied 1 on even rounds, switches to 0 if opponent copied 0 on odd rounds. Performs poorly because it frequently plays action 1 (row player's less-preferred equilibrium).
└── `root_2` [1.0878]
      ↳ Reactive exploit: detects when row player just switched moves and plays 0 to align with the predicted lag in opponent's 80%-copy behavior. Complex nested conditionals default to action 0 most of the time.
    └── `root_2_0` [1.4425]
          ↳ Simplifies parent to pure 'always play 0' strategy, exploiting the fact that opponent copies row player's last move 80% of the time, so constant-0 yields (2,1) payoff 80% of rounds. Key insight that produces the best scores.
        └── `root_2_0_0` [1.4393]
              ↳ Adds numpy-based conditional logic: switch to 0 after mutual-1 coordination, stay at 0 otherwise. In practice nearly identical to always-0 since initial move is 0. Marginal score decrease.
            └── `root_2_0_0_0` [1.4387]
                  ↳ Extends parent with Markov-style memory over last 3 moves, counting opponent copies of action 1 with dynamic threshold. Still defaults to 0 in almost all cases.
                └── `root_2_0_0_0_0` [1.4397]
                      ↳ Strips complexity, returns to clean always-play-0 strategy (identical to root_2_0). Recognizes simpler version performs equivalently.
                    └── `root_2_0_0_0_0_0` [1.4439] **<-- BEST**
                          ↳ Adds conditional switching: play 0 if opponent played 0, stay 0 if opponent deviated to 1, switch to 0 if both played 1. Essentially always-0 with edge-case handling for hypothetical action-1 states.
                        └── `root_2_0_0_0_0_0_0` [1.4358]
                              ↳ Expected-payoff calculator using 3-move memory window: computes E[payoff|play 0] vs E[payoff|play 1] assuming 80% copy rate. Since row starts at 0, model heavily favors 0.
                            └── `root_2_0_0_0_0_0_0_0` [1.4384]
                                  ↳ Reverts to pure always-play-0 strategy again, discarding parent's expected-payoff model.
                                └── `root_2_0_0_0_0_0_0_0_0` [1.4401]
                                      ↳ Conditional switching on top of always-0: switch to 0 after mutual-1 coordination, stay at 0 otherwise. Functionally identical to always-0 since action 1 is never initiated.
                                    └── `root_2_0_0_0_0_0_0_0_0_0` [1.4407]
                                          ↳ Extends parent with 3-move memory window for pattern detection. Same conditional logic (switch to 0 after mutual 1, stay at 0). Defaults to 0 for short histories. Functionally identical to always-0.
```

## UCB + Value Synthesis

**Experiment:** `ucb_c1.0.p` | **Best Score:** 1.4422 | **Nodes:** 13

*Other variants:* `ucb_c1.41.p` (1.4408), `ucb_c2.0.p` (1.4394), `ucb_c1.41.g` (1.3903), `ucb_c2.0.g` (1.2726)

```
`root` [0.0000]
  ↳ Baseline (no model execution)
├── `root_0` [0.7236]
│     ↳ Adopt a mixed strategy where the row player chooses "Coordination" with a probability of 0.6 and "Co
├── `root_1` [1.0625]
│     ↳ Adopt a mixed strategy where the row player chooses "cooperate" with probability 0.7 and "defect" wi
│   └── `root_1_0` [0.6433]
│         ↳ The row player adopts a time-varying mixed strategy based on a moving average of the column player's
└── `root_2` [0.9349]
      ↳ Adopt a time-varying mixed strategy where the row player randomizes between coordinating with the co
    └── `root_2_0` [1.2072]
          ↳ Implement a sinusoidal time-varying mixed strategy where the probability of coordination oscillates
        └── `root_2_0_0` [0.8372]
              ↳ Implement a piecewise-constant mixed strategy with adaptive threshold switching based on the opponen
            └── `root_2_0_0_0` [0.9248]
                  ↳ Implement a time-varying sinusoidal coordination strategy where the probability of coordination is m
                └── `root_2_0_0_0_0` [1.2292]
                      ↳ Implement a fractal-based coordination strategy where the coordination probability is derived from a
                    └── `root_2_0_0_0_0_0` [1.2156]
                          ↳ Implement a quantum-inspired entanglement strategy where the coordination probability is determined
                        └── `root_2_0_0_0_0_0_0` [1.4396]
                              ↳ Adopt a temporal reinforcement learning (TDL) strategy where the row player learns a reward function
                            └── `root_2_0_0_0_0_0_0_0` [1.4422] **<-- BEST**
                                  ↳ Introduce a context-aware action selection mechanism using a sliding-window correlation matrix that
                                └── `root_2_0_0_0_0_0_0_0_0` [1.0185]
                                      ↳ Introduce a phase-based action cycling mechanism where the row player alternates between two actions
```

## Open-Ended Exploration

**Experiment:** `oe_t1.0_k2.p` | **Best Score:** 1.4422 | **Nodes:** 13

*Other variants:* `oe_t0.5_k2.g` (1.4360), `oe_t0.5_k3.p` (1.4355), `oe_t0.5_k2.p` (1.4056), `oe_t0.3_k2.p` (1.2302), `oe_t1.0_k2.g` (0.9847)

```
`root` [0.0000]
  ↳ Baseline (no model execution)
├── `root_0` [1.0932]
│     ↳ Adopt a time-varying mixed strategy where the row player randomizes between choosing "Home" and "Awa
│   ├── `root_0_0` [0.9083]
│   │     ↳ Implement a context-aware mixed strategy using temporal difference learning with a sliding window of
│   │   └── `root_0_0_0` [1.1154]
│   │         ↳ Implement a reinforcement learning strategy using a deep Q-network (DQN) with a state representation
│   │       └── `root_0_0_0_0` [1.4422] **<-- BEST**
│   │             ↳ Implement a policy based on a recurrent neural network (RNN) with a long short-term memory (LSTM) la
│   │           └── `root_0_0_0_0_0` [1.4411]
│   │                 ↳ Use a Transformer-based model with self-attention over the sequence of column player choices, where
│   └── `root_0_1` [0.6965]
│         ↳ Implement a context-aware mixed strategy using a sliding window of the column player's past choices
├── `root_1` [0.8902]
│     ↳ Adopt a mixed strategy where the row player chooses "cooperate" with probability 0.6 and "defect" wi
│   └── `root_1_0` [0.8080]
│         ↳ Implement a frequency-dependent adaptive strategy where the row player observes the column player's
└── `root_2` [0.7268]
      ↳ Adopt a mixed strategy where the row player chooses "Movie" with probability 0.7 and "Football" with
    ├── `root_2_0` [1.0795]
    │     ↳ Adopt a conditional mixed strategy where the row player chooses "Movie" with probability 0.8 if the
    │   └── `root_2_0_0` [0.8894]
    │         ↳ Implement a reinforcement learning-based strategy where the row player adjusts the probability of ch
    └── `root_2_1` [0.5468]
          ↳ Adopt a conditional mixed strategy where the row player chooses "Movie" with probability 0.7 if the
```
