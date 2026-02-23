# Best Trees: Qwen3-4B on Titanic (accuracy)

## AIRA (best variant)

**Experiment:** `aira_evo.g` | **Best Score:** 0.8828 | **Nodes:** 13

*Other variants:* `aira_mcts.g` (0.8732), `aira_greedy.g` (0.8612)

```
`root` [0.7655]
  ↳ Baseline: runs the provided starter code.
├── `root_0` [FAIL]
│     ↳ RandomForest (100 trees, max_depth=10) with FamilySize, LabelEncoded Sex/Embarked/Cabin, median-imputed Age/Fare. Failed because LabelEncoder hit unseen Cabin values in test set, and the heredoc was truncated so the fix was never written.
├── `root_1` [0.8493]
│     ↳ RandomForest (100 trees, max_depth=10) with manual dict-based encoding to avoid unseen-label issues. Engineered FamilySize, IsAlone, AgeBand (5 bins), FareBand (4 quartiles), Cabin first-letter, Pclass*Age interaction. Age imputed by Pclass-group median. First attempts with XGBoost/LightGBM/CatBoost ensemble failed on dtype errors, fell back to RF-only.
│   ├── `root_1_0` [FAIL]
│   │     ↳ [crossover] Attempted RF+XGBoost ensemble (100 trees each, lr=0.1, max_depth=6) with one-hot encoding and median imputation. Heredoc was consistently truncated mid-line preventing test prediction code from being written -- failed across 8 rewrite attempts.
│   └── `root_1_1` [0.8804]
│         ↳ [crossover] Averaged RF (100 trees, max_depth=10) and XGBoost (100 trees, lr=0.1, max_depth=6) predictions. Simpler feature set than parent: just Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone with LabelEncoder. Dropped AgeBand/FareBand/Cabin.
├── `root_2` [0.7871]
│     ↳ Ensemble of RF+XGBoost+LightGBM+CatBoost with Title extraction from Name (Mr/Mrs/Miss/Master/Rare), FamilySize, IsAlone, Cabin filled with mode. CatBoost initially failed; second rewrite used RF+XGB+LGBM only. Lower score likely from poor Cabin imputation (mode vs first-letter).
├── `root_3` [0.8565]
│     ↳ RandomForest (100 trees, max_depth=10) with custom AgeGroup function (Child/Teen/Young/Adult/Senior), FamilySize, LabelEncoded Sex/Embarked/AgeGroup. Clean 3-step execution on full training set.
└── `root_4` [0.8828] **<-- BEST**
      ↳ Ensemble of RF+XGBoost+LightGBM with sklearn Pipeline, ColumnTransformer, SelectKBest(f_classif, k=10). Extensive features: FamilySize, AgeBand (5 bins), FareBand (4 bins), Title extraction with detailed mapping (Mlle->Miss, Mme->Mrs). Multiple debug rounds for dtype issues; final version used 19 features. Best depth-1 node.
    ├── `root_4_0` [0.8421]
    │     ↳ Simplified parent to single XGBoost (100 trees, max_depth=6) with FamilySize and custom AgeGroup (6 bins). LabelEncoded Sex/Embarked/Pclass, median imputation. Dropped the ensemble, Title, and FareBand features -- score regressed.
    │   ├── `root_4_0_0` [FAIL]
    │   │     ↳ Tried to add Cabin one-hot dummies (first letter extraction with pd.get_dummies) and IsAlone feature on top of parent's XGBoost. Heredoc was truncated at 'test_cabin_dum' preventing complete code across 8 rewrite attempts.
    │   └── `root_4_0_1` [FAIL]
    │         ↳ Added LabelEncoded Cabin (full string, not first letter) to parent's XGBoost model. Heredoc consistently truncated preventing complete code generation across 8 attempts.
    ├── `root_4_1` [0.8278]
    │     ↳ Ensemble of RF+XGBoost+LightGBM (default params) with FamilySize, IsAlone, AgeGroup (5 bins), FareGroup (4 bins), LabelEncoded Sex/Embarked. Dropped Name/Ticket/Cabin entirely. Lower score than parent due to removing Cabin/Title features.
    └── `root_4_2` [FAIL]
          ↳ Single XGBoost (100 trees, max_depth=6, lr=0.1, subsample=0.8) with FamilySize, AgeGroup (6 bins), LabelEncoded Sex/Embarked/Cabin. Heredoc truncated at train_test_split line across 8 attempts.
```

## UCB + Value Synthesis

**Experiment:** `ucb_c1.0.p` | **Best Score:** 0.9426 | **Nodes:** 13

*Other variants:* `ucb_c1.41.g` (0.9402), `ucb_c1.41.p` (0.9019), `ucb_c2.0.g` (0.8876), `ucb_c2.0.p` (0.8804)

```
`root` [0.0000]
  ↳ Baseline (no model execution)
├── `root_0` [0.7727]
│     ↳ Use a gradient-boosting model (e.g., XGBoost or LightGBM) with feature engineering derived from cate
│   └── `root_0_0` [0.7871]
│         ↳ Replace the gradient-boosting model with a neural network (e.g., a multilayer perceptron) using a de
│       ├── `root_0_0_0` [0.9306]
│       │     ↳ Replace the neural network with a support vector machine (SVM) using a radial basis function (RBF) k
│       │   └── `root_0_0_0_0` [FAIL]
│       │         ↳ Replace the SVM with a gradient boosting machine (e.g., XGBoost) and apply target encoding to catego
│       └── `root_0_0_1` [0.7775]
│             ↳ Introduce a learning rate scheduling strategy (e.g., reduce the learning rate by a factor of 0.1 eve
│           └── `root_0_0_1_0` [0.8612]
│                 ↳ Replace the neural network architecture with a gradient-boosted tree model (e.g., XGBoost or LightGB
│               └── `root_0_0_1_0_0` [0.6459]
│                     ↳ Apply a deep autoencoder-based feature learning approach to extract latent representations from the
│                   └── `root_0_0_1_0_0_0` [0.6244]
│                         ↳ Replace the autoencoder with a self-supervised contrastive learning framework (e.g., SimCLR or MoCo)
│                       └── `root_0_0_1_0_0_0_0` [0.9426] **<-- BEST**
│                             ↳ Apply a transformer-based model with tokenized and embedded categorical features (e.g., using embedd
│                           └── `root_0_0_1_0_0_0_0_0` [FAIL]
│                                 ↳ Replace the transformer architecture with a gradient-boosted tree ensemble (e.g., XGBoost or LightGB
├── `root_1` [FAIL]
│     ↳ Use a gradient-boosting model (e.g., XGBoost or LightGBM) with categorical features encoded via targ
└── `root_2` [FAIL]
      ↳ Use a gradient boosting machine (e.g., XGBoost) with target encoding for categorical variables like
```

## Open-Ended Exploration

**Experiment:** `oe_t0.3_k2.p` | **Best Score:** 0.9785 | **Nodes:** 13

*Other variants:* `oe_t1.0_k2.g` (0.9641), `oe_t0.5_k3.p` (0.9569), `oe_t0.5_k2.p` (0.8565), `oe_t0.5_k2.g` (0.8517), `oe_t1.0_k2.p` (0.8086)

```
`root` [0.0000]
  ↳ Baseline (no model execution)
├── `root_0` [FAIL]
│     ↳ Apply a gradient-boosting ensemble (e.g., XGBoost or LightGBM) with targeted feature interactions, s
├── `root_1` [0.9139]
│     ↳ Use a gradient boosting model (e.g., XGBoost) with targeted feature engineering such as creating int
│   ├── `root_1_0` [0.8230]
│   │     ↳ Replace the gradient boosting model with a neural network (e.g., a multi-layer perceptron) and incor
│   │   └── `root_1_0_0` [0.8493]
│   │         ↳ Replace the neural network with a random forest ensemble, but instead of using raw features, apply r
│   │       └── `root_1_0_0_0` [0.5861]
│   │             ↳ Replace the random forest ensemble with a gradient boosting machine (e.g., XGBoost or LightGBM) and
│   │           └── `root_1_0_0_0_0` [0.8469]
│   │                 ↳ Replace the gradient boosting model with a neural network (e.g., a multi-layer perceptron) and intro
│   │               └── `root_1_0_0_0_0_0` [0.8612]
│   │                     ↳ Replace the neural network with a random forest ensemble, using bootstrap aggregating (bagging) to i
│   │                   └── `root_1_0_0_0_0_0_0` [0.5861]
│   │                         ↳ Replace the random forest ensemble with a gradient boosting machine (GBM) such as XGBoost or LightGB
│   │                       └── `root_1_0_0_0_0_0_0_0` [FAIL]
│   │                             ↳ Replace the gradient boosting model with a neural network architecture using a shallow feedforward n
│   └── `root_1_1` [0.9785] **<-- BEST**
│         ↳ Replace the gradient boosting model with a neural network (e.g., a multi-layer perceptron) and use a
│       └── `root_1_1_0` [0.8612]
│             ↳ Replace the neural network with a random forest ensemble, and instead of polynomial features, apply
└── `root_2` [FAIL]
      ↳ Use a gradient boosting machine (e.g., XGBoost or LightGBM) with target encoding for categorical var
```
