import json
import sys
from pathlib import Path

import uvicorn

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))
project_path = src_path.parent


from typing import List

import pandas as pd
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

from utils.load_params import load_params
from alibi_detect.saving import load_detector
import os
import datetime
from sqlalchemy import create_engine

app = FastAPI()
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ['DATABASE_URL'].replace('postgres://', 'postgresql://')

params = load_params(params_path='params.yaml')

model_path = project_path / params.model_output.dir / params.model_output.filename
feat_cols = params.features.cols

min_batch_size = params.drift_detect.min_batch_size

cd = load_detector(Path('models')/'drift_detector')
model = load(filename=model_path)



class Customer(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class Request(BaseModel):
    data: List[Customer]

@app.post("/predict")
async def predict(info: Request = Body(..., example={
    "data": [
        {
            "CreditScore": 619,
            "Age": 42,
            "Tenure": 2,
            "Balance": 0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
        },
        {
            "CreditScore": 699,
            "Age": 39,
            "Tenure": 21,
            "Balance": 0,
            "NumOfProducts": 2,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 93826.63
        }
    ]
})):
    json_list = json.loads(info.json())
    data = json_list['data']
    input_data = pd.DataFrame(data)
    probs = model.predict_proba(input_data)[:,0]
    probs = probs.tolist()
    collect_batch(json_list) 
    return probs

@app.get("/drift_data")
async def get_drift_data():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        sql_query = "SELECT * FROM p_val_table"
        df_p_val = pd.read_sql(sql_query, con=conn)
    engine.dispose()
    parsed = json.loads(df_p_val.to_json())
    return json.dumps(parsed) 

def collect_batch(json_list, batch_size_thres = min_batch_size, batch = []):
    data = json_list['data']
    for req_json in data:
        batch.append(req_json)
    L = len(batch)
    if L >= batch_size_thres:
        X = pd.DataFrame.from_records(batch)
        preds = cd.predict(X)
        p_val = preds['data']['p_val']
        now = datetime.datetime.now()
        data = [[now] + p_val.tolist()]
        columns = ['time'] + feat_cols
        df_p_val = pd.DataFrame(data=data, columns=columns)
        print('Writing to database')
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            df_p_val.to_sql('p_val_table', con=conn, if_exists='append', index=False)
        engine.dispose()
        batch.clear()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)