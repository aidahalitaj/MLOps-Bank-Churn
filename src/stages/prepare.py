import sys
import argparse
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
project_path = src_path.parent
sys.path.append(str(src_path))

import os

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split



from utils.load_params import load_params
def get_project_paths():
    return project_path, src_path

def load_and_merge_data(raw_dir, countries, file_pattern):
    paths = [raw_dir/file_pattern.format(country=c)  for c in countries]
    df = pd.concat([pd.read_csv(p) for p in paths])
    return df

def split_data(df, features, target,test_size,random_state):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def save_data(X_train, X_test, y_train, y_test, project_path, split_paths, processed_dir):
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_pickle(project_path / split_paths.X_train)
    X_test.to_pickle(project_path / split_paths.X_test)
    y_train.to_pickle(project_path / split_paths.y_train)
    y_test.to_pickle(project_path / split_paths.y_test)


def prepare_data(config_path):
    project_path, _ = get_project_paths()
    params = load_params(config_path)

    raw_dir = project_path / params.data.raw_dir
    processed_dir = project_path / params.outputs.processed_data_dir

    df = load_and_merge_data(raw_dir, params.data.countries, params.data.file_pattern)
    X_train, X_test, y_train, y_test = split_data(df, params.features.cols, params.target, params.split.test_size, params.random_state)
    save_data(X_train, X_test, y_train, y_test, project_path, params.outputs.split_data, processed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", required=True, help="Path to params.yaml")
    args = parser.parse_args()
    prepare_data(config_path=Path(args.config))