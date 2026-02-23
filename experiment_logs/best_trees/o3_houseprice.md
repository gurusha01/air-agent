# Best Trees: o3 on House Price (R²)

## AIRA (best variant)

*No experiments available*

## UCB + Value Synthesis

*No experiments available*

## Open-Ended Exploration

**Experiment:** `o3_oe.g` | **Best Score:** 0.8800 | **Nodes:** 13

```
`root` [0.8800] **<-- BEST**
  ↳ Baseline (no model execution)
├── `root_0` [FAIL]
│     ↳ Train a LightGBM regressor with extensive feature engineering: (1) apply a Yeo-Johnson or log1p tran
├── `root_1` [FAIL]
│     ↳ Build a two–level stacking ensemble: (1) train diverse first-stage models—ElasticNet on log-transfor
├── `root_2` [FAIL]
│     ↳ Train a single CatBoostRegressor that natively handles the ~40 categorical columns, leaving them in
├── `root_3` [FAIL]
│     ↳ Variation #1 – LightGBM “deep-leaf + noise” tweak  
Stay with a single LightGBM model but (a) increa
├── `root_4` [FAIL]
│     ↳ Variation: Keep the same LightGBM model but (1) raise max_depth from -1 (no limit) to 8, (2) lower l
├── `root_5` [FAIL]
│     ↳ Baseline recap:
• Pipeline: simple preprocessing (median imputation, one-hot encode categoricals) →
├── `root_6` [FAIL]
│     ↳ LightGBM “slow-learn, many trees” refinement  
Step-by-step  
1. Keep the same LightGBM model but:
├── `root_7` [FAIL]
│     ↳ Variation: Seed-bagged XGBoost with slightly lower learning rate and heavier column/row subsampling
├── `root_8` [FAIL]
│     ↳ LightGBM-based baseline tweak: 
• Keep the same LightGBM learner but (a) apply a log1p transformatio
├── `root_9` [FAIL]
│     ↳ Refine the current XGBoost-based pipeline by adding two focused tweaks:

1. Extended skew handling
├── `root_10` [FAIL]
│     ↳ Variation: Keep the current LightGBM-based pipeline but tighten the preprocessing and regularization
└── `root_11` [FAIL]
      ↳ Variation: keep the LightGBM model but introduce leave-one-out target encoding (with Gaussian smooth
```
