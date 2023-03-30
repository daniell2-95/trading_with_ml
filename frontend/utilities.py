import streamlit as st
import pandas as pd
import requests
import os
import yaml

with open(os.path.dirname(__file__) + '/config.yaml', 'r') as file:
	config = yaml.safe_load(file)
    
API_SERVER = config["API_SERVER"]

def load_mlflow_runs() -> pd.DataFrame:

    """
    Loads MLflow run data from the API server.

    Args:
        None

    Returns:
        pd.DataFrame: A pandas dataframe containing the MLflow run data.
    """

    uri = API_SERVER + "/load_mlruns"
    response = requests.get(uri)
    if response.status_code == 200:
        return pd.read_json(response.text)
    else:
        print("Failed to get mlflow runs:", response.status_code)
        return None

def make_predictions(model_name: str, data: pd.DataFrame) -> list:

    """
    Makes predictions using a specified model and input data.

    Args:
        model_name: The name of the model to use for predictions.
        data: The input data for the model, as a pandas dataframe.

    Returns:
        list: A list of predictions returned by the API server.
    """

    api_url = f"{API_SERVER}/predict"
    data_dict = data.to_dict()
    response = requests.post(api_url, json = {"model_name": model_name, "data": data_dict})
    if response.status_code == 200:
        return response.json()["predictions"]
    else:
        print(f"Request failed with code {response.status_code}: {response.text}")
        return None

@st.cache
def load_data(ticker: str) -> pd.DataFrame:

    """
    Helper function to retrieve OHLC data using the ticker symbol

    Parameters:
        ticker: str representing stock ticker

    Returns:
        pandas dataframe containing OHLC data
    """

    data = requests.post(API_SERVER + "/load", json = {'ticker': ticker})
    return pd.read_json(data.json())

@st.cache
def get_dates(data: pd.DataFrame) -> pd.Series:

    """
    Helper function to retrieve dates from OHLC data

    Parameters:
        data: pandas dataframe containing OHLC price data and
              various engineered features

    Returns:
        series of sorted dates
    """

    return data['t'].sort_values(ascending = False)

def filter_df_by_dates(stock_data: pd.DataFrame, 
                       start_date: pd.datetime, 
                       end_date: pd.datetime) -> pd.DataFrame:
    
    """
    Helper function to filter stock data by start and end dates

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        start_date: str representing start date
        end_date: str representing end date

    Returns:
        filtered pandas dataframe by start and end date
    """

    return stock_data[(stock_data['t'] >= start_date) & (stock_data['t'] <= end_date)]