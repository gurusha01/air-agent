# Best Trees: o3 on Titanic (accuracy)

## AIRA (best variant)

**Experiment:** `o3_aira_evo.g` | **Best Score:** 0.9785 | **Nodes:** 13

*Other variants:* `o3_aira_mcts.g` (0.9211), `o3_aira_greedy.g` (0.8947)

```
`root` [0.7655]
  ↳ Baseline: runs the provided starter code.
├── `root_0` [0.8589]
│     ↳ XGBClassifier with Title extraction (rare titles grouped), FamilySize, IsAlone, CabinLetter, TicketPrefix. Age imputed by Title-group median, one-hot encoding. First attempt crashed on pandas groupby bug; second fixed with .transform().
│   └── `root_0_0` [0.8828]
│         ↳ [crossover] Ensemble of XGBClassifier + GradientBoosting + RandomForest with soft-voting and StratifiedKFold CV. Adds Age imputation by Title+Pclass group, StandardScaler on numerics. Improved preprocessing over parent.
│       └── `root_0_0_0` [0.8756]
│             ↳ [crossover] Adds TicketFreq (ticket group size), HasCabin indicator, FarePP (fare per person) to parent's ensemble. Uses GBM+RF+LogReg+XGB. Three failed attempts (pandas bugs, NaN in GBM), fourth fixed with row-wise Age imputation fallback.
├── `root_1` [0.8660]
│     ↳ XGBClassifier in sklearn Pipeline with ColumnTransformer+OneHotEncoder. Title extraction with Officer/Royal grouping, FamilySize, IsAlone, CabinLetter, TicketPrefix. More sophisticated title categorization than root_0.
├── `root_2` [0.9785] **<-- BEST**
│     ↳ CatBoostClassifier with native categorical handling. Title extraction, FamilySize, IsAlone, TicketPrefix, CabinLetter. Age imputed by Title median. Simplest approach but by far the best Titanic score -- CatBoost's native categoricals outperformed all one-hot/label encoding approaches.
│   ├── `root_2_0` [0.8397]
│   │     ↳ Switched CatBoost to XGBClassifier with Pipeline, added FarePerPerson, log1p(Fare), first-char ticket prefix. Lost CatBoost's native categorical advantage. Massive regression.
│   ├── `root_2_1` [0.8780]
│   │     ↳ Switched CatBoost to StackingClassifier (RF+GBM base, LogReg meta) with StandardScaler and FarePerPerson. One-hot encoded instead of native categoricals. Significant regression.
│   │   └── `root_2_1_0` [0.8445]
│   │         ↳ Switched stacking back to single XGBClassifier with 5-fold StratifiedKFold CV. Title grouped to Mr/Mrs/Miss/Master/Rare, CabinDeck, TicketPrefix. Further regression.
│   └── `root_2_2` [0.8421]
│         ↳ Kept CatBoostClassifier but added TicketGroupSize, FarePerPerson, more detailed title mapping. Despite keeping CatBoost, score regressed massively from parent's 0.978 -- likely different hyperparameters or overfitting from extra features.
├── `root_3` [0.8708]
│     ↳ XGBClassifier with GridSearchCV-style params, Title mapped to Royal/Officer groups, FamilySize, IsAlone, CabinLetter, TicketPrefix.
│   └── `root_3_0` [0.8469]
│         ↳ Switched XGB to CatBoostClassifier with native categorical handling (Sex, Embarked, Title, CabinDeck, TicketPrefix). Score regressed from parent's XGB.
└── `root_4` [0.8589]
      ↳ GradientBoostingClassifier in Pipeline with FarePerPerson, TicketPrefix. First attempt used LGBMClassifier but timed out (3600s); second hit OneHotEncoder API change; third fixed with GBM and sparse_output=False.
```

## UCB + Value Synthesis

**Experiment:** `o3_ucb.g` | **Best Score:** 0.9306 | **Nodes:** 13

```
`root` [0.7655]
  ↳ Baseline (no model execution)
├── `root_0` [0.8421]
│     ↳ Implement a gradient-boosted decision tree model (e.g., LightGBM) with rich, hand-crafted categorica
│   └── `root_0_0` [0.8780]
│         ↳ Train a “wide-and-deep” neural network that learns dense embeddings for every categorical token inst
│       └── `root_0_0_0` [0.8541]
│             ↳ Gradient-Boosted Decision Trees with Fold-Wise Target Encoding: 1) Engineer family-level features (e
│           └── `root_0_0_0_0` [0.8517]
│                 ↳ Strategy: Construct a passenger–passenger graph and perform semi-supervised graph learning.  
1. Nod
│               └── `root_0_0_0_0_0` [0.6507]
│                     ↳ CatBoost-based “family-aware target encoding” stack.  
1. Hand-craft relational aggregate features i
├── `root_1` [0.9091]
│     ↳ Train a gradient-boosted decision-tree model (e.g., LightGBM) after building richer categorical/cont
│   └── `root_1_0` [0.9163]
│         ↳ Deploy a Wide-&-Deep neural network that jointly learns (a) a sparse “wide” linear component with L1
│       └── `root_1_0_0` [0.6411]
│             ↳ Construct an explicit passenger-relation graph and train a Graph Neural Network (GNN):  
• Nodes = p
└── `root_2` [0.8373]
      ↳ Strategy: Train a LightGBM gradient-boosted decision tree with rich categorical embeddings and group
    └── `root_2_0` [0.9306] **<-- BEST**
          ↳ Strategy: Build a passenger-relationship graph and train a Graph Neural Network (GNN).

1. Graph con
        └── `root_2_0_0` [0.8995]
              ↳ Strategy — Group-aware Set Transformer with Multiple-Instance Learning (MIL)

1. Data restructuring
            └── `root_2_0_0_0` [0.8684]
                  ↳ Strategy: Text-augmented Gradient Boosting Ensemble

1. Rich text features  
   • Feed the raw “Name
```

## Open-Ended Exploration

**Experiment:** `o3_oe.g` | **Best Score:** 0.9593 | **Nodes:** 13

```
`root` [0.7655]
  ↳ Baseline (no model execution)
├── `root_0` [0.9522]
│     ↳ Train a CatBoostClassifier (gradient-boosted decision trees that natively handle categorical data) u
│   └── `root_0_0` [0.7703]
│         ↳ Adopt a self-supervised TabNet pipeline:  
1) Pre-train TabNet with a masked‐feature reconstruction
│       └── `root_0_0_0` [0.8756]
│             ↳ Construct a passenger-relation graph and train a Graph Neural Network (GNN).  
1) Nodes = individual
│           └── `root_0_0_0_0` [0.5287]
│                 ↳ Hierarchical-Bayesian mixed-effects modelling:  
• Build a logistic-regression likelihood but introd
├── `root_1` [0.8589]
│     ↳ Engineer socially-oriented features (family_size, group_ticket_survival_rate, cabin_deck, passenger_
│   └── `root_1_0` [0.7416]
│         ↳ Graph-Neural-Network passenger network:  
1. Build a heterogeneous graph where each node is a passen
└── `root_2` [0.9187]
      ↳ Train a CatBoostClassifier that natively handles the categorical variables (Sex, Embarked, Ticket pr
    └── `root_2_0` [0.8565]
          ↳ Train an entity-embedding neural network: 
1. Pre-process: fill missing Age/Fare with KNN imputation
        └── `root_2_0_0` [0.8900]
              ↳ Stacked generalisation ensemble:  
1. Generate out-of-fold (OOF) prediction columns for 6 very diffe
            └── `root_2_0_0_0` [0.8732]
                  ↳ Exploit the relational structure in the manifest with a graph-neural network.  
1. Build an undirect
                └── `root_2_0_0_0_0` [0.7703]
                      ↳ Strategy: Hierarchical Bayesian Logistic Regression with Partial-Pooling Group Effects
  
  1. Defin
                    └── `root_2_0_0_0_0_0` [0.9593] **<-- BEST**
                          ↳ Strategy: Relational Graph Attention Network (R-GAT) over the passenger interaction graph  

1. Buil
```
