#!/bin/sh

# 🟦 Random Forest hyperparameter sweep
dvc exp run \
  -S model.name=random_forest \
  -S models.random_forest.n_estimators=range(50, 201, 50) \
  -S models.random_forest.max_depth=range(5, 21, 5) \
  --queue

# 🟧 Logistic Regression sweep
dvc exp run \
  -S model.name=logistic_regression \
  -S models.logistic_regression.C=range(0.1, 1.1, 0.3) \
  --queue

# 🟩 Decision Tree sweep
dvc exp run \
  -S model.name=decision_tree \
  -S models.decision_tree.max_depth=range(4, 13, 4) \
  --queue

# 🟨 SVM sweep
dvc exp run \
  -S model.name=svm \
  -S models.svm.C=range(0.1, 1.1, 0.3) \
  --queue

# 🟥 XGBoost sweep
dvc exp run \
  -S model.name=xgboost \
  -S models.xgboost.n_estimators=range(50, 151, 50) \
  -S models.xgboost.max_depth=range(3, 10, 3) \
  -S models.xgboost.learning_rate=range(0.01, 0.11, 0.03) \
  --queue
