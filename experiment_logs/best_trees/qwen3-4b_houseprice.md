# Best Trees: Qwen3-4B on House Price (R²)

## AIRA (best variant)

**Experiment:** `aira_greedy.g` | **Best Score:** 0.9212 | **Nodes:** 13

*Other variants:* `aira_mcts.g` (0.9203), `aira_evo.g` (0.9066)

```
`root` [0.8800]
  ↳ Baseline: runs the provided starter code.
├── `root_0` [FAIL]
│     ↳ RandomForest (100 trees) in sklearn Pipeline with OneHotEncoder(drop='first', handle_unknown='ignore') and median/mode imputation. Repeatedly failed due to KeyError: 'Id' -- dropped Id column early then tried to reference test['Id'] for submission DataFrame. Never fixed across 8 attempts.
├── `root_1` [0.9128]
│     ↳ Ensemble of RF (500 trees, max_depth=10, min_samples_split=5), GradientBoosting (500 trees, lr=0.05, max_depth=6, subsample=0.8), XGBoost (500 trees), and LightGBM (500 trees) with SimpleImputer and OneHotEncoder. Clean execution, best single-depth ensemble approach.
├── `root_2` [FAIL]
│     ↳ Attempted RF/GBR/XGBoost/LightGBM ensemble with manual OneHotEncoder and TotalSF/Age feature engineering. Failed due to pd.concat TypeError (numpy vs DataFrame mismatch) then KeyError: 'Id'. Never produced submission.
├── `root_3` [0.8869]
│     ↳ RandomForest (100 trees) in sklearn Pipeline with SimpleImputer and OneHotEncoder. No feature engineering beyond basic imputation and encoding. Minimal approach that slightly beats baseline.
└── `root_4` [0.9212] **<-- BEST**
      ↳ Ensemble of RF+XGBoost+LightGBM+CatBoost with 10 engineered features: TotalSF (1stFlr+2ndFlr+LowQual), TotalBsmtSF, TotalPorchSF, YearBuilt-YearRemodAdd interaction, Age (2023-YearBuilt), GarageArea_per_Cars, TotalBath (all bath types summed), GrLivArea_per_Bedroom, OverallQual_squared, OverallQual_log. Best node.
    ├── `root_4_0` [0.8848]
    │     ↳ Simplified to single RandomForest (100 trees, max_depth=10, min_samples_split=5) in Pipeline. Dropped all parent's feature engineering and ensemble. Score regressed significantly.
    ├── `root_4_1` [0.9098]
    │     ↳ Single XGBoost (1000 trees, lr=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0) in Pipeline. No custom features -- relies on XGBoost's regularization and higher tree count. Dropped ensemble but strong tuning.
    ├── `root_4_2` [0.8859]
    │     ↳ Ensemble of RF+XGBoost+LightGBM with GridSearchCV, OneHotEncoder(handle_unknown='ignore'). Dropped parent's engineered features. Slightly worse than parent.
    ├── `root_4_3` [0.8818]
    │     ↳ XGBoost with custom interaction features (TotalArea=1stFlr+2ndFlr+GrLivArea, BsmtAreaRatio, GarageAreaRatio, PorchArea). Initially failed on raw object-dtype categoricals; fixed with OneHotEncoder pipeline.
    ├── `root_4_4` [0.9129]
    │     ↳ Single XGBoost (1000 trees, lr=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0) in Pipeline. Nearly identical to root_4_1 with slightly different random state. Second-best score.
    ├── `root_4_5` [FAIL]
    │     ↳ Attempted ensemble of RF+XGBoost+LightGBM+CatBoost with StandardScaler and pd.get_dummies. Failed due to SyntaxError -- pd.get_dummies not valid as sklearn transformer in Pipeline. Never produced submission.
    └── `root_4_6` [0.8869]
          ↳ [debug] Fixed root_4_5's failures by falling back to simple RandomForest (100 trees) in Pipeline. Abandoned broken ensemble approach entirely.
```

## UCB + Value Synthesis

**Experiment:** `ucb_c1.0.p` | **Best Score:** 0.9193 | **Nodes:** 13

*Other variants:* `ucb_c2.0.g` (0.9149), `ucb_c1.41.p` (0.9121), `ucb_c1.41.g` (0.9057), `ucb_c2.0.p` (0.8973)

```
`root` [0.0000]
  ↳ Baseline (no model execution)
├── `root_0` [0.4988]
│     ↳ Apply polynomial feature expansion to the input variables (e.g., square, cube, or interaction terms
│   ├── `root_0_0` [0.9045]
│   │     ↳ Apply a gradient boosting machine (e.g., XGBoost or LightGBM) with tree-based splits and early stopp
│   │   └── `root_0_0_0` [FAIL]
│   │         ↳ Replace the gradient boosting model with a neural network (e.g., a multi-layer perceptron) using con
│   └── `root_0_1` [0.8432]
│         ↳ Apply polynomial feature expansion with degree 3 and include interaction terms (e.g., GrLivArea * To
│       └── `root_0_1_0` [0.9094]
│             ↳ Replace the polynomial feature expansion with a gradient boosting machine (e.g., XGBoost or LightGBM
│           └── `root_0_1_0_0` [0.9191]
│                 ↳ Apply a neural network with a deep architecture (e.g., 3–5 hidden layers with 64–128 neurons each) u
│               └── `root_0_1_0_0_0` [0.9193] **<-- BEST**
│                     ↳ Apply a gradient boosting ensemble (e.g., XGBoost, LightGBM, or CatBoost) with careful tuning of tre
│                   └── `root_0_1_0_0_0_0` [0.8865]
│                         ↳ Apply a random forest ensemble with bootstrap aggregating (bagging) and introduce synthetic features
│                       └── `root_0_1_0_0_0_0_0` [0.9094]
│                             ↳ Replace the random forest ensemble with a lightweight gradient boosting model (e.g., XGBoost or Ligh
│                           └── `root_0_1_0_0_0_0_0_0` [0.9094]
│                                 ↳ Introduce a neural network with a shallow architecture (2–3 hidden layers, 64 neurons each) using a
├── `root_1` [FAIL]
│     ↳ Apply a gradient boosting model (e.g., XGBoost or LightGBM) with cross-validation to capture non-lin
└── `root_2` [FAIL]
      ↳ Apply polynomial feature expansion to capture non-linear relationships in the dataset. Transform key
```

## Open-Ended Exploration

**Experiment:** `oe_t0.5_k3.p` | **Best Score:** 0.9201 | **Nodes:** 13

*Other variants:* `oe_t0.5_k2.p` (0.9169), `oe_t1.0_k2.g` (0.9156), `oe_t1.0_k2.p` (0.9141), `oe_t0.3_k2.p` (0.9128), `oe_t0.5_k2.g` (0.9036)

```
`root` [0.0000]
  ↳ Baseline (no model execution)
├── `root_0` [0.8188]
│     ↳ Apply polynomial feature expansion to the existing input variables (e.g., square footage, number of
│   ├── `root_0_0` [0.9089]
│   │     ↳ Replace Ridge regression with a Gradient Boosting Machine (e.g., XGBoost or LightGBM) to capture com
│   │   └── `root_0_0_0` [0.8680]
│   │         ↳ Apply a Random Forest ensemble with bootstrap aggregation (bagging) to improve robustness and reduce
│   │       └── `root_0_0_0_0` [0.9156]
│   │             ↳ Replace the Random Forest ensemble with a Gradient Boosting Machine (GBM) such as XGBoost, using a f
│   │           └── `root_0_0_0_0_0` [0.9082]
│   │                 ↳ Introduce a neural network with a deep architecture (e.g., 3–5 hidden layers with 128–256 neurons ea
│   │               └── `root_0_0_0_0_0_0` [0.9102]
│   │                     ↳ Replace the neural network with a gradient boosting ensemble (e.g., XGBoost, LightGBM, or CatBoost)
│   │                   └── `root_0_0_0_0_0_0_0` [0.9056]
│   │                         ↳ Introduce a deep residual network (ResNet-inspired architecture) with skip connections, using fully
│   │                       └── `root_0_0_0_0_0_0_0_0` [0.9056]
│   │                             ↳ Replace the deep residual network with a gradient-boosted tree ensemble (e.g., XGBoost or LightGBM)
│   │                           └── `root_0_0_0_0_0_0_0_0_0` [FAIL]
│   │                                 ↳ Replace the gradient-boosted tree ensemble with a neural network architecture using a mixture of ful
│   └── `root_0_1` [0.9201] **<-- BEST**
│         ↳ Apply a Gradient Boosting Machine (e.g., XGBoost or LightGBM) with early stopping and tuned hyperpar
├── `root_1` [FAIL]
│     ↳ Apply polynomial feature expansion on key numerical variables (e.g., area, year_built) with degree 2
└── `root_2` [FAIL]
      ↳ Apply polynomial feature expansion on key numerical predictors (e.g., square footage, number of bedr
```
