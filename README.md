# E2E ML project using open-source MLOps tools
This repository is my implementation of an end-to-end ML project using open-source MLOps tools, based on Alexis’ O’Reilly tutorial: https://www.oreilly.com/live-events/open-source-mlops-in-4-weeks/0636920080215/.

I used the tutorial as a starting point for the overall structure, but I refactored and reorganised parts of the codebase to make it clearer and more maintainable, and to better match how I would set up a real project.

**Branches and project progression**

I kept the work split by week to show how the project evolved step by step:

- week-1 → initial setup and baseline approach
- week-2 → incremental improvements and organisation changes
- week-3 → further iterations and experimentation
- week-4 → the most complete version of the project

If you want to see the final state, use the week-4 branch.

## Problem Description and Dataset
This dataset contains 10,000 records, each of which corresponds to a different bank's user. The target is `Exited`, a binary variable that describes whether the user decided to leave the bank. There are row and customer identifiers, four columns describing personal information about the user (surname, location, gender and age), and some other columns containing information related to the loan (such as credit score, the current balance in the user's account and whether they are an active member among others).

Dataset source: https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann

## Use Case
The objective is to train an ML model that returns the probability of a customer churning. This is a binary classification task, therefore F1-score is a good metric to evaluate the performance of this dataset as it weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.


## Setup
Python 3.8+ is required to run code from this repo.
```
$ git clone https://github.com/aidahalitaj/bank-churn-mlops.git
$ cd bank-churn-mlops
$ virtualenv -p python3 .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## What's inside
```bash
$ tree 
.
├── Churn_Modelling_France.csv # raw data file
├── Churn_Modelling_Spain.csv # raw data file
├── LICENSE
├── README.md # this file
├── TODO.md # describes next steps
├── TrainChurnModel.ipynb # jupyter notebook with model training code
├── clf-model.joblib # saved model
└── feat_imp.csv # table with feature importance values
```
