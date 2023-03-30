from model_experimentation.experiment_run import run_experiment
from dtypes import Ticker, ExperimentParametersXGBoost, ExperimentParametersAlgoMA, ModelRequest

import os
import yaml
import boto3
import mlflow
import uvicorn
import pandas as pd

# FastAPI libray
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Initiate app instance
app = FastAPI(title = 'Stock Direction + Volitility Model Training and Predictions', 
              version = '1.0',
              description = 'API to train and retrieve predictions')

# Load config
with open(os.path.dirname(__file__) + '/config.yaml', 'r') as file:
	config = yaml.safe_load(file)

# ENV variables
AWS_S3_BUCKET = config["AWS_S3_BUCKET"]
AWS_ACCESS_KEY_ID = config["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = config["AWS_SECRET_ACCESS_KEY"]
FINNHUB_API_KEY = config["FINNHUB_API_KEY"]
MLFLOW_TRACKING_URI = config["MLFLOW_TRACKING_URI"]

# s3 client
s3 = boto3.client('s3',
                  aws_access_key_id = AWS_ACCESS_KEY_ID, 
                  aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
    
# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    return {'message': 'System is healthy'}

# ML API endpoint for loading data
@app.post("/load")
def load(ticker: Ticker) -> JSONResponse:
    """
    Endpoint to load data 
    """
    response = s3.get_object(Bucket = AWS_S3_BUCKET, Key = f"prices/{ticker.ticker}.csv")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        data = pd.read_csv(response.get("Body"))
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

    return JSONResponse(data.to_json())

# ML API endpoint for loading data
@app.get("/load_mlruns")
def load_mlruns() -> JSONResponse:
    """
    Endpoint to load mlruns experiments table
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.search_runs()

# ML API endpoint for predicting data
@app.post("/predict")
def predict(model_request: ModelRequest):
    """
    Endpoint to make predictions based on model name and data to make predictions on
    """
    # Load model from MLflow registry
    model = mlflow.pyfunc.load_model(model_uri = f'models:/{model_request.model_name}/1')
    
    # Make predictions on input data
    preds = model.predict(pd.DataFrame.from_dict(model_request.data))
    return {"predictions": preds.tolist()}

# ML API endpoint for training and returning training results
@app.post("/experiment_ML_xgboost")
async def experiment_xgboost(parameters: ExperimentParametersXGBoost) -> None:
    args = {
        "ticker": parameters.ticker,
        "target": parameters.target,
        "model_type": parameters.model_type,
        "model_class": parameters.model_class,
        "name": parameters.name,
        "current_date": parameters.current_date,
        "features": parameters.features,
        "horizon" : parameters.horizon,
        "window" : parameters.window,
        "max_evals" : parameters.max_evals,
        "scores" : parameters.scores,
        "parallelism" : parameters.parallelism,
        "max_depth": parameters.max_depth,
        "n_estimators": parameters.n_estimators,
        "reg_alpha": parameters.reg_alpha,
        "max_delta_step": parameters.max_delta_step,
        "min_split_loss": parameters.min_split_loss,
        "learning_rate": parameters.learning_rate,
        "min_child_weight": parameters.min_child_weight,
        "scale_pos_weight": parameters.scale_pos_weight
    }
    run_experiment(args)

@app.post("/experiment_algo_moving_average")
async def experiment_algo_MA(parameters: ExperimentParametersAlgoMA) -> None:
    args = {
        "ticker": parameters.ticker,
        "target": parameters.target,
        "model_type": parameters.model_type,
        "model_class": parameters.model_class,
        "name": parameters.name,
        "current_date": parameters.current_date,
        "features": parameters.features,
        "horizon" : parameters.horizon,
        "window" : parameters.window,
        "max_evals" : parameters.max_evals,
        "scores" : parameters.scores,
        "parallelism" : parameters.parallelism,
        "short_term_ma" : parameters.short_term_ma,
        "long_term_ma": parameters.long_term_ma
    }
    run_experiment(args)

if __name__ == '__main__':
    uvicorn.run("main:app", host = "127.0.0.1", port = 5001, reload = True)