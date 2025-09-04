import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
project_path = src_path.parent
sys.path.append(str(src_path))

import argparse
import pandas as pd
from alibi_detect.cd import TabularDrift
from alibi_detect.saving import save_detector
from joblib import load
from utils.load_params import load_params

def train_drift_detector(params):
    processed_data_dir = project_path / params.outputs.processed_data_dir
    model_dir = project_path / params.model_output.dir
    model_path = model_dir / params.model_output.filename

    model = load(model_path)

    # X_test = pd.read_pickle(processed_data_dir/'X_test.pkl')
    # X_train = pd.read_pickle(processed_data_dir/'X_train.pkl')
    # X = pd.concat([X_test, X_train])
    X = pd.read_pickle(processed_data_dir/'X_train.pkl')

    feat_names = X.columns.tolist()
    preprocessor = model[:-1]
    cd = TabularDrift(X, 
                    p_val=.05, 
                    preprocess_fn=preprocessor.transform)

    detector_path = model_dir/'drift_detector'
    detector_path.mkdir(parents=True, exist_ok=True)
    save_detector(cd, detector_path)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params = load_params(params_path=args.config)
    train_drift_detector(params)