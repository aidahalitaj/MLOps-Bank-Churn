# ChurnPredict — Bank Customer Churn Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![DVC](https://img.shields.io/badge/DVC-pipeline-yellow.svg)](https://dvc.org)
[![Deployed on Fly.io](https://img.shields.io/badge/deployed-Fly.io-purple.svg)](https://fly.io)

End-to-end MLOps project covering the full lifecycle: data pipeline, model training, experiment tracking, API serving, interactive frontend, CI/CD, cloud deployment, and data drift monitoring.

**[Live Demo →](https://churn-prediction-app.fly.dev)**

![ChurnPredict UI](docs/screenshot.png)

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | Python, scikit-learn, pandas |
| Pipeline | DVC (data versioning + orchestration) |
| API | FastAPI, Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Monitoring | Alibi Detect (TabularDrift), PostgreSQL |
| Deployment | Docker, Fly.io |
| CI/CD | GitHub Actions |

## Features

- **Interactive prediction UI** — reactive sliders for each feature, instant churn probability
- **Sample customer profiles** — pre-filled templates to demo different risk levels
- **What-if analysis** — tweak individual features and see how predictions change
- **Risk factor breakdown** — shows which features contribute most to the prediction
- **Data drift monitoring** — statistical tests on incoming data, stored in PostgreSQL

## Project Structure

```
├── src/
│   ├── app/
│   │   ├── main.py              # FastAPI app, endpoints, drift collection
│   │   └── static/              # Frontend (HTML, CSS, JS)
│   ├── stages/
│   │   ├── prepare.py           # Data loading, feature selection, train/test split
│   │   ├── train.py             # Model training
│   │   ├── evaluate.py          # Metrics, feature importance, confusion matrix
│   │   └── drift_detector.py    # Train Alibi Detect drift detector
│   └── utils/
│       └── load_params.py       # Config loader
├── data/
│   ├── raw.dvc                  # DVC-tracked raw data
│   └── processed/               # Train/test splits (generated)
├── models/
│   ├── clf-model.joblib         # Trained model
│   └── drift_detector/          # Serialised drift detector
├── reports/
│   ├── metrics.json             # F1, ROC AUC
│   ├── feat_imp.csv             # Feature importance scores
│   └── figures/cm.png           # Confusion matrix plot
├── notebooks/                   # Exploration & testing notebooks
├── .github/workflows/
│   ├── train-model.yaml         # Retrain on data/code changes
│   └── deploy-api.yaml          # Deploy on tag push
├── dvc.yaml                     # Pipeline definition
├── params.yaml                  # All model & pipeline config
├── Dockerfile                   # Production container
├── fly.toml                     # Fly.io deployment config
└── requirements.txt             # Dev dependencies
```

## ML Pipeline

Four DVC stages, fully reproducible with `dvc repro`:

```
prepare → train → evaluate → train_drift_detector
```

| Stage | What it does |
|-------|-------------|
| `prepare` | Loads raw CSVs (France, Spain), selects features, splits train/test |
| `train` | Trains the active model on processed data |
| `evaluate` | Computes metrics, feature importance, confusion matrix |
| `train_drift_detector` | Fits an Alibi Detect TabularDrift detector on training data |

### Model Options

Configured in `params.yaml` — switch by changing `model.name`:

| Model | Status |
|-------|--------|
| Random Forest | **Active** (n_estimators=200, max_depth=10) |
| Logistic Regression | Available |
| Decision Tree | Available |
| SVM (RBF) | Available |
| XGBoost | Available |

### Current Metrics

| Metric | Score |
|--------|-------|
| F1 Score | 0.54 |
| ROC AUC | 0.87 |

### Feature Importance

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Age | 0.231 |
| 2 | NumOfProducts | 0.188 |
| 3 | IsActiveMember | 0.133 |
| 4 | Balance | 0.027 |
| 5 | Tenure | 0.018 |
| 6 | CreditScore | 0.008 |
| 7 | HasCrCard | -0.003 |
| 8 | EstimatedSalary | -0.003 |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the interactive frontend |
| `POST` | `/predict_single` | Single customer prediction — returns churn probability and label |
| `POST` | `/predict` | Batch prediction — accepts a list of customers, returns probabilities |
| `GET` | `/drift_data` | Returns drift p-values from PostgreSQL |

**Example — single prediction:**

```bash
curl -X POST https://churn-prediction-app.fly.dev/predict_single \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":619,"Age":42,"Tenure":2,"Balance":0,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":101348.88}'
```

## Local Setup

```bash
git clone https://github.com/aidahalitaj/bank-churn-mlops.git
cd bank-churn-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull data and models
dvc pull

# Run the server
uvicorn src.app.main:app --host 0.0.0.0 --port 8080
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

## Deployment

The app runs on **Fly.io** in a Docker container (London region, shared CPU, 1 GB RAM).

Machines auto-stop when idle and auto-start on incoming requests — zero cost when not in use.

### CI/CD

Two GitHub Actions workflows:

- **`train-model`** — triggers on push to `main` when data, pipeline stages, or params change. Runs `dvc exp run`, pushes artifacts, and posts a metrics report as a commit comment.
- **`deploy`** — triggers on tag push matching `deploy-v*`. Pulls the model with DVC, builds the Docker image, and deploys to Fly.io.

```bash
# Deploy a new version
git tag deploy-v1.0.0
git push origin deploy-v1.0.0
```

## Drift Monitoring

Incoming prediction requests are collected in batches (minimum 50). When a batch fills, **Alibi Detect's TabularDrift** (Kolmogorov-Smirnov test) runs per-feature statistical tests comparing the batch against training data. The resulting p-values are stored in PostgreSQL and available via the `/drift_data` endpoint.

## Dataset

[Kaggle — Deep Learning A-Z: ANN](https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann) — 10,000 bank customer records with 8 features and a binary target (`Exited`).

Features: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

## Author

**Aida Halitaj** — [aidahalitaj.github.io](https://aidahalitaj.github.io)
