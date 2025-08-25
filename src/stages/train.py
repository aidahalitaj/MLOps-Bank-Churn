
import sys
import argparse
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
# src_path = /workspaces/open-source-mlops-e2e-starting-point/src
project_path = src_path.parent
# project_path = /workspaces/open-source-mlops-e2e-starting-point/
sys.path.append(str(src_path))


from utils.load_params import load_params
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# import seaborn as sns
import pandas as pd
# import matplotlib.pyplot as plt

def get_project_paths():
    return project_path, src_path

def load_training_data(params, project_path):
    X_train_path = project_path / params.outputs.split_data.X_train
    y_train_path = project_path / params.outputs.split_data.y_train
    X_train = pd.read_pickle(X_train_path)
    y_train = pd.read_pickle(y_train_path)
    return X_train, y_train


def build_model(params):
    model_name = params.model.name
    model_params = params.models[model_name]
    random_state = params.random_state


    if model_name == "random_forest":
        clf = RandomForestClassifier(random_state=random_state,
                                 n_estimators=model_params.n_estimators,
                                 max_depth=model_params.max_depth)

    elif model_name == "logistic_regression":
        clf = LogisticRegression(
            random_state=random_state,
            penalty=model_params.penalty,
            C=model_params.C,
            solver=model_params.solver
        )

    elif model_name == "decision_tree":
        clf = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=model_params.max_depth,
            criterion=model_params.criterion
        )

    elif model_name == "svm":
        clf = SVC(
            probability=True,  # Needed for predict_proba
            kernel=model_params.kernel,
            C=model_params.C,
            gamma=model_params.gamma,
            random_state=random_state
        )

    elif model_name == "xgboost":
        clf = XGBClassifier(
            n_estimators=model_params.n_estimators,
            max_depth=model_params.max_depth,
            learning_rate=model_params.learning_rate,
            subsample=model_params.subsample,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state
        )

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return Pipeline(steps=[("preprocessor", SimpleImputer()), ("clf", clf)])


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)

def save_model(model, project_path, model_dir, model_filename):
    model_dir_path = project_path / model_dir
    model_dir_path.mkdir(parents=True, exist_ok=True)

    model_path = model_dir_path / model_filename
    dump(model, model_path)

def train_pipeline(config_path):
    project_path, _ = get_project_paths()
    params = load_params(config_path)
    X_train, y_train = load_training_data(params, project_path)
    model = build_model(params)
    trained_model = train_model(model, X_train, y_train)
    save_model(trained_model, project_path, params.model_output.dir, params.model_output.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        required=True, help="Path to params.yaml")
    args = parser.parse_args()
    train_pipeline(config_path=Path(args.config))
