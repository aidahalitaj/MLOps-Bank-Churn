from joblib import load
from eli5.sklearn import PermutationImportance
from sklearn.metrics import make_scorer
import eli5
from sklearn.metrics import (confusion_matrix, f1_score, make_scorer,
                             roc_auc_score)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from train import train_model
from utils.load_params import load_params
import sys
import argparse
from pathlib import Path
import json

src_path = Path(__file__).parent.parent.resolve()
project_path = src_path.parent
sys.path.append(str(src_path))


def get_project_paths():
    return project_path, src_path


# Loading model + test data
def load_evaluation_data(params, project_path):
    X_test_path = project_path / params.outputs.split_data.X_test
    y_test_path = project_path / params.outputs.split_data.y_test
    X_test = pd.read_pickle(X_test_path)
    y_test = pd.read_pickle(y_test_path)
    return X_test, y_test

# Making predictions


def predict(model, X_test, threshold):
    y_prob = model.predict_proba(X_test)
    y_pred = y_prob[:, 1] >= threshold
    return y_pred, y_prob

# Evaluating metrics (F1, ROC AUC)


def compute_metrics(y_test, y_pred, y_prob):
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    return {"f1_score": f1, "roc_auc": roc_auc}

def save_metrics(metrics: dict, output_path: Path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

def save_confusion_matrix(y_test, y_pred, fig_path):
    fig_path.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
    output_path = fig_path / 'cm.png'
    plt.savefig(output_path)
    plt.close()

# Calculating feature importance


def compute_feature_importance(model, X_test, y_test, feat_cols, random_state):
    out_feat_names = model[:-1].get_feature_names_out(feat_cols)
    preprocessor = model.named_steps['preprocessor']
    clf = model.named_steps['clf']
    X_test_transformed = preprocessor.transform(X_test)
    perm = PermutationImportance(clf, scoring=make_scorer(
        f1_score), random_state=random_state).fit(X_test_transformed, y_test)
    eli5.show_weights(perm, feature_names=out_feat_names)
    feat_imp = zip(X_test.columns.tolist(), perm.feature_importances_)
    df_feat_imp = pd.DataFrame(feat_imp,
                               columns=['feature', 'importance'])
    df_feat_imp = df_feat_imp.sort_values(by='importance', ascending=False)
    return df_feat_imp

# Saving feature importance


def save_feature_importance(df_feat_imp, out_path):
    return df_feat_imp.to_csv(out_path, index=False)


def evaluate_pipeline(config_path):
    project_path, _ = get_project_paths()
    params = load_params(config_path)

    model_path = project_path / params.model.output_dir / params.model.filename
    model = load(model_path)
    X_test, y_test = load_evaluation_data(params, project_path)

    threshold = params.evaluation.threshold
    y_pred, y_prob = predict(model, X_test, threshold)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics_path = project_path / params.outputs.metrics_file
    save_metrics(metrics, metrics_path)

    fig_path = project_path / params.outputs.figures_dir
    save_confusion_matrix(y_test, y_pred, fig_path)

    feat_cols = params.features.cols
    random_state = params.random_state
    df_feat_imp = compute_feature_importance(
        model, X_test, y_test, feat_cols, random_state)
    out_path = project_path / params.outputs.feature_importance_csv
    save_feature_importance(df_feat_imp, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        required=True, help="Path to params.yaml")
    args = parser.parse_args()
    evaluate_pipeline(config_path=Path(args.config))
