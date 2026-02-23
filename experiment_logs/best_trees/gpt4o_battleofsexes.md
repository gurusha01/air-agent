# Best Trees: GPT-4o on Battle of Sexes (score)

## AIRA (best variant)

**Experiment:** `gpt4o_aira_evo.g` | **Best Score:** 1.4432 | **Nodes:** 13

*Other variants:* `gpt4o_aira_greedy.g` (1.4431), `gpt4o_aira_mcts.g` (1.4394)

```
`root` [1.0227]
  ↳ Baseline: runs the provided starter code.
├── `root_0` [0.9681]
│     ↳ Reactive to opponent's last move with 80% probability of matching. Slightly worse than baseline because matching opponent's move isn't always optimal for row player.
├── `root_1` [1.3584]
│     ↳ Coordination-detection strategy: stays on action 0 if both coordinated on 0 last round; otherwise plays 0 with 80% probability. Strong 0-bias produces winning formula.
│   └── `root_1_0` [0.6384]
│         ↳ Deterministic version of root_1: always switches to 1 on mismatch instead of 80/20 random. Removing randomness destroys performance.
├── `root_2` [1.3584]
│     ↳ Nearly identical to root_1: stay on 0 if matched last round, else 80/20 toward 0. Same coordination-detection approach.
│   ├── `root_2_0` [1.3559]
│   │     ↳ Trivial refactor of root_2 with slightly different variable naming. Essentially identical strategy.
│   └── `root_2_1` [1.4054]
│         ↳ Simplified opponent-reactive: always play 0 if opponent played 0, play 0 with 90% probability if opponent played 1. Very strong 0-bias.
├── `root_3` [1.1902]
│     ↳ Stay on 0 if matched, else uniform random (50/50). Weaker than the 80/20 variants due to higher chance of playing action 1.
│   ├── `root_3_0` [1.3615]
│   │     ↳ Improvement over root_3: opponent-reactive with 80% action 0 on mismatch instead of 50/50.
│   │   └── `root_3_0_0` [1.4432] **<-- BEST**
│   │         ↳ [crossover] Best node. Deterministic parity-based: plays 0 on even rounds, repeats own last move on odd rounds. Simple but effective alternation pattern.
│   └── `root_3_1` [1.0206]
│         ↳ Tit-for-tat on mismatch: copies opponent's last move when they disagree. Near baseline performance since copying opponent doesn't help row player.
└── `root_4` [1.3633]
      ↳ Same 80/20 coordination strategy as root_1/root_2 with different syntax.
    └── `root_4_0` [0.9187]
          ↳ Inverts probability: 80% toward action 1 on mismatch instead of action 0. Heavily penalized since coordinating on 0 is dominant.
```

## UCB + Value Synthesis

**Experiment:** `gpt4o_ucb.g` | **Best Score:** 1.4401 | **Nodes:** 13

```
`root` [1.0227]
  ↳ Baseline (no model execution)
├── `root_0` [1.3173]
│     ↳ Implement a Reinforcement Learning strategy using Q-Learning. This approach will allow the row playe
│   └── `root_0_0` [1.4372]
│         ↳ Consider implementing a Multi-Armed Bandit (MAB) approach to tackle the Battle of Sexes game. The pl
│       └── `root_0_0_0` [1.1715]
│             ↳ Implement a Q-learning strategy, which is a model-free reinforcement learning algorithm. In this app
│           └── `root_0_0_0_0` [0.9367]
│                 ↳ Consider implementing a Deep Q-Network (DQN) to enhance the adaptability and performance over the cl
├── `root_1` [1.0722]
│     ↳ Evaluate using a Deep Q-Network (DQN) approach, where the row player is modeled as an agent in a rei
│   └── `root_1_0` [1.0780]
│         ↳ Augment the Deep Q-Network with a multi-agent reinforcement learning approach, specifically using a
│       └── `root_1_0_0` [1.0735]
│             ↳ Adopt a Generative Adversarial Network (GAN) approach for training the row player. Implement a setup
│           └── `root_1_0_0_0` [1.0776]
│                 ↳ Incorporate a reinforcement learning-based approach using a Deep Q-Network (DQN) for the row player.
│               └── `root_1_0_0_0_0` [1.0763]
│                     ↳ Experiment with a multi-agent reinforcement learning (MARL) approach where multiple agents play the
└── `root_2` [1.3428]
      ↳ Implement a reinforcement learning approach where the row player uses Q-learning to dynamically adju
    └── `root_2_0` [1.4401] **<-- BEST**
          ↳ Implement a Bayesian learning approach where the row player uses a Bayesian updating mechanism to dy
        └── `root_2_0_0` [0.8216]
              ↳ One alternative approach to improve the score is to implement a Multi-Armed Bandit (MAB) framework u
```

## Open-Ended Exploration

**Experiment:** `gpt4o_oe.g` | **Best Score:** 1.4428 | **Nodes:** 13

```
`root` [1.0227]
  ↳ Baseline (no model execution)
├── `root_0` [0.5299]
│     ↳ Implement a Q-learning approach to adaptively learn the optimal strategy over time. By treating the
├── `root_1` [0.7753]
│     ↳ Implement a Reinforcement Learning (RL) agent using a Q-learning approach. Develop the agent to adju
└── `root_2` [0.8284]
      ↳ Use a reinforcement learning (RL) approach, such as Q-learning, where the row player is treated as a
    └── `root_2_0` [0.8300]
          ↳ Consider implementing a multi-agent deep reinforcement learning (MADRL) approach. In this setup, tre
        └── `root_2_0_0` [1.4387]
              ↳ Implement a genetic algorithm (GA) approach, treating each player as a population of potential strat
            └── `root_2_0_0_0` [0.8486]
                  ↳ Implement a reinforcement learning approach using Q-learning, focusing on optimizing action-selectio
                └── `root_2_0_0_0_0` [1.4403]
                      ↳ To enhance the performance of the Battle of Sexes task, explore the implementation of a genetic algo
                    └── `root_2_0_0_0_0_0` [1.4379]
                          ↳ Introduce a deep reinforcement learning approach using a neural network-based model. Employ a multi-
                        └── `root_2_0_0_0_0_0_0` [1.4395]
                              ↳ Explore a genetic algorithm (GA) approach to optimize the row player's strategy. In this setup, enco
                            └── `root_2_0_0_0_0_0_0_0` [1.4373]
                                  ↳ Consider implementing a Bayesian Optimization approach to maximize the row player's payoff. This tec
                                └── `root_2_0_0_0_0_0_0_0_0` [1.4428] **<-- BEST**
                                      ↳ Implement a Reinforcement Learning (RL) approach using Q-learning. This strategy involves formulatin
                                    └── `root_2_0_0_0_0_0_0_0_0_0` [1.4395]
                                          ↳ Consider implementing a Deep Q-Learning Network (DQN) to leverage the advantages of deep learning fo
```
