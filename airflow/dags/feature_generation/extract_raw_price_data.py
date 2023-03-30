#!/usr/bin/env python3

from datetime import date

import finnhub
import pandas as pd
import yaml
import os
import boto3
import json

def extract_and_store_raw_price_data(client: finnhub.client.Client,
                                     ticker: str,
                                     interval: str,
                                     start_date: str,
                                     end_date: str):
    """
    This function extracts raw price data for a stock from Finnhub API 
    for a specified time interval, stores it in AWS S3 and prints status messages.

    Parameters:
        client: a finnhub.client.Client object, the client object for accessing the Finnhub API
        ticker: a string, the stock ticker of the stock to extract data for
        interval: a string, the time interval for the data (i.e. 'D', 'W', 'M')
        start_date: a string in the format 'YYYY-MM-DD', the start date of the data to extract
        end_date: a string in the format 'YYYY-MM-DD', the end date of the data to extract
        
    Returns:
        None
    """
    print(f"Extracting raw {ticker} price data...")

    ohlcv_api_call = \
        client.stock_candles(ticker, 
                             interval, 
                             int(pd.Timestamp(start_date).timestamp()), 
                             int(pd.Timestamp(end_date).timestamp()))

    s3.put_object(Bucket = AWS_S3_BUCKET, 
                  Body = json.dumps(ohlcv_api_call), 
                  Key = f'prices/{stock}.json')
    
    print(f"Done extracting raw {ticker} price data...")


if __name__ == "__main__":
    # load config
    TODAY = date.today()
    with open(os.path.dirname(__file__) + '/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    AWS_S3_BUCKET = config['AWS_S3_BUCKET']
    AWS_ACCESS_KEY_ID = config['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = config['AWS_SECRET_ACCESS_KEY']
    FINNHUB_API_KEY = config['FINNHUB_API_KEY']

    s3 = boto3.client('s3', 
                      aws_access_key_id = AWS_ACCESS_KEY_ID, 
                      aws_secret_access_key = AWS_SECRET_ACCESS_KEY)

    for stock in config['s_and_p_500'][:11]:

        extract_and_store_raw_price_data(client = finnhub.Client(api_key = FINNHUB_API_KEY),
                                         ticker = stock,
                                         interval = 'D', 
                                         start_date = '1900-01-01', 
                                         end_date = TODAY)