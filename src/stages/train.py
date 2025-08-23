
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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
    n_estimators = params.model.random_forest.n_estimators
    max_depth = params.model.random_forest.max_depth
    random_state = params.random_state
    clf = RandomForestClassifier(random_state=random_state,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth)
    model = Pipeline(
        steps=[("preprocessor", SimpleImputer()), ("clf", clf)])
    return model


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
    save_model(trained_model, project_path, params.model.output_dir, params.model.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        required=True, help="Path to params.yaml")
    args = parser.parse_args()
    train_pipeline(config_path=Path(args.config))
