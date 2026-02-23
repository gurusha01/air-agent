# Best Trees: GPT-4o on Titanic (accuracy)

## AIRA (best variant)

**Experiment:** `gpt4o_aira_evo.g` | **Best Score:** 0.9306 | **Nodes:** 13

*Other variants:* `gpt4o_aira_greedy.g` (0.9234), `gpt4o_aira_mcts.g` (0.8923)

```
`root` [0.7655]
  ↳ Baseline: runs the provided starter code.
├── `root_0` [0.8086]
│     ↳ Vanilla RandomForest with basic features only (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked). Median imputation for Age/Fare, LabelEncoded Sex/Embarked. No feature engineering.
├── `root_1` [0.8923]
│     ↳ VotingClassifier of RF+GradientBoosting+LogisticRegression with FamilySize feature. Soft voting ensemble with engineered FamilySize = SibSp + Parch + 1.
│   └── `root_1_0` [0.8828]
│         ↳ [crossover] Two RandomForest models in VotingClassifier. Simpler ensemble than parent, dropped GBM and LogReg.
├── `root_2` [0.9163]
│     ↳ RandomForest with GridSearchCV for hyperparameter tuning plus FamilySize feature. Cross-validated parameter search over n_estimators and max_depth.
│   └── `root_2_0` [0.9306] **<-- BEST**
│         ↳ RandomForest with Title extraction from Name column (Mr/Mrs/Miss/Master with rare title grouping). Title feature was the single most impactful engineering decision. Best node.
│       ├── `root_2_0_0` [0.6746]
│       │     ↳ Added Cabin_assigned binary feature but lost the Title feature in the rewrite. Catastrophic regression due to losing Title.
│       ├── `root_2_0_1` [0.9067]
│       │     ↳ RF with GridSearchCV plus Title + FamilySize features in a complex pipeline. More pipeline overhead but same core features as parent.
│       │   └── `root_2_0_1_0` [0.9067]
│       │         ↳ RF (200 trees, max_depth=7) with Title extracted via str.split() instead of regex. Same score as parent -- Title feature still present.
│       └── `root_2_0_2` [0.8900]
│             ↳ RF + GradientBoosting probability-averaged ensemble with shallow max_depth=5.
├── `root_3` [0.8469]
│     ↳ Simple RandomForest with integer-encoded Embarked (S=0, C=1, Q=2). No feature engineering.
│   └── `root_3_0` [0.8852]
│         ↳ [crossover] RF + GradientBoosting VotingClassifier with basic encoding.
└── `root_4` [0.9187]
      ↳ RF + LogisticRegression + SVC VotingClassifier with StandardScaler. Trained on full training set (no validation split).
```

## UCB + Value Synthesis

**Experiment:** `gpt4o_ucb.g` | **Best Score:** 0.9211 | **Nodes:** 13

```
`root` [0.7655]
  ↳ Baseline (no model execution)
├── `root_0` [0.8301]
│     ↳ One approach to improving the Titanic survival prediction score is to focus on feature engineering b
│   └── `root_0_0` [0.8828]
│         ↳ Experiment with ensemble learning techniques such as stacking, bagging, or boosting to combine the s
│       └── `root_0_0_0` [0.9211]
│             ↳ Implement a deep learning approach using neural networks, which can automatically capture complex pa
│           └── `root_0_0_0_0` [0.7679]
│                 ↳ A possible strategy to improve the Titanic survival prediction score could be to use ensemble method
├── `root_1` [FAIL]
│     ↳ To improve the Titanic survival prediction score, consider implementing a gradient boosting model su
└── `root_2` [0.8349]
      ↳ Consider implementing a feature engineering strategy by creating additional features related to pass
    └── `root_2_0` [0.9211] **<-- BEST**
          ↳ Another strategy to enhance the survival prediction score could be implementing a machine learning m
        └── `root_2_0_0` [0.8995]
              ↳ One viable strategy to improve the Titanic survival prediction score is to apply a deep learning app
            └── `root_2_0_0_0` [0.8254]
                  ↳ To improve the Titanic survival prediction score, consider adopting a decision tree-based ensemble m
                └── `root_2_0_0_0_0` [0.9019]
                      ↳ Consider using a neural network-based approach, such as a multi-layer perceptron (MLP), to classify
                    └── `root_2_0_0_0_0_0` [0.9067]
                          ↳ Consider using an ensemble approach by combining the predictions of multiple diverse models to impro
                        └── `root_2_0_0_0_0_0_0` [0.8995]
                              ↳ To improve the score, consider implementing a deep learning approach using a neural network, which c
```

## Open-Ended Exploration

**Experiment:** `gpt4o_oe.g` | **Best Score:** 0.9713 | **Nodes:** 13

```
`root` [0.7655]
  ↳ Baseline (no model execution)
├── `root_0` [0.8134]
│     ↳ Implement a Random Forest classifier and conduct feature engineering to improve model performance. I
├── `root_1` [0.9091]
│     ↳ One strategy to improve the score on the Titanic survival prediction task is to use ensemble methods
│   └── `root_1_0` [0.9091]
│         ↳ A promising strategy to further improve the Titanic survival prediction score is to utilize gradient
│       └── `root_1_0_0` [0.9115]
│             ↳ Explore deep learning techniques by implementing a neural network model for the Titanic survival pre
│           └── `root_1_0_0_0` [0.9258]
│                 ↳ To improve the score, shift focus towards ensemble methods by implementing a stacked ensemble model.
│               └── `root_1_0_0_0_0` [0.9426]
│                     ↳ Explore deep learning approaches by implementing a neural network model. Begin with a simple feedfor
│                   └── `root_1_0_0_0_0_0` [0.9713] **<-- BEST**
│                         ↳ Consider using ensemble learning techniques to improve the prediction accuracy. A random forest clas
│                       └── `root_1_0_0_0_0_0_0` [0.8923]
│                             ↳ Consider implementing a Gradient Boosting Machine (GBM) approach. GBMs are effective for binary clas
│                           └── `root_1_0_0_0_0_0_0_0` [0.9402]
│                                 ↳ Consider exploring ensemble techniques by combining multiple diverse model types to capitalize on th
│                               └── `root_1_0_0_0_0_0_0_0_0` [0.9330]
│                                     ↳ Consider utilizing domain-specific feature engineering to potentially enhance the score further. Beg
└── `root_2` [0.8900]
      ↳ One strategy to improve the Titanic survival prediction score is to employ ensemble methods such as
    └── `root_2_0` [0.7823]
          ↳ Explore feature engineering by creating new features from existing ones to capture more information
```
