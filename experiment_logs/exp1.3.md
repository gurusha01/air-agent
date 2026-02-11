Pi 5 vs Pi 105 Trajectory Comparision                                                                                                                                     
                                                                                                                                                            
Metrics                                                                                                                                                                 
┌──────────────────────────────┬────────┬────────┐                                                                                                              
│            Metric            │   pi5  │  pi105 │                                                                                                 
├──────────────────────────────┼────────┼────────┤                                                                                                               
│ Final score                  │ 0.8804 │ 0.8780 │                                                                                                               
├──────────────────────────────┼────────┼────────┤
│ Improvement                  │ 15.0%  │ 14.7%  │
├──────────────────────────────┼────────┼────────┤
│ Peak score during trajectory │ 0.8804 │ 0.8971 │
├──────────────────────────────┼────────┼────────┤
│ Unique scripts written       │ 4      │ 7      │
├──────────────────────────────┼────────┼────────┤
│ Validations                  │ 7      │ 8      │
└──────────────────────────────┴────────┴────────┘
---
pi 5 Strategy: "Fix data cleaning"

- Only used RandomForest(n_estimators=100) - never changed hyperparameters
- Focused on Embarked missing values (fillna, row filtering)
- One refactoring change with no functional impact
- Got stuck re-running the same script; final improvement likely from RF non-determinism

---
pi 105 Strategy: "Incremental hyperparameter tuning"

- Systematic one-variable-at-a-time tuning: max_depth → min_samples_split → max_features → bootstrap
- Tried n_estimators 100→200→300→200 (reverted when it regressed)
- Fixed pandas deprecation warnings proactively
- Used train median for test Age imputation (better practice)
- Hit 0.8971 peak but couldn't reproduce it after overshooting with n_estimators=300
---

Diversity Assessment

pi 105 shows more diverse actions, but only marginally. It explored 5 hyperparameter dimensions vs pi₅'s zero. However, both trajectories share critical limitations:

- No model diversity — both used only RandomForest, never tried GradientBoosting, XGBoost, Logistic Regression, or ensembles
- No feature engineering — neither extracted Title from Name (classic high-value Titanic feature), Deck from Cabin, or interaction features
- No cross-validation — neither used CV for model selection
- Repetitive tail behavior — both got stuck repeating the same script at the end

---
Key Takeaway

RL training from pi 5 -> pi 105 taught the agent to do incremental hyperparameter tuning (absent in pi 5), but it reinforced a narrow "tweak RF hyperparameters" strategy rather
than encouraging broader exploration across model families or feature engineering approaches. This supports the need for diversity-aware rewards — the current reward
signal rewards any improvement but doesn't incentivize exploring fundamentally different strategies.
