#!/usr/bin/env python3
from io import StringIO
import finnhub
import pandas as pd
import yaml
import os
import boto3

def extract_and_store_raw_financials_data(client: finnhub.client.Client,
                                          ticker: str):

    """
    Extracts quarterly financial data for a given stock ticker using the Finnhub API and stores the data in a CSV file
    on an AWS S3 bucket.

    Parameters:
        client: A client object for accessing the Finnhub API.
        ticker: The stock ticker for which to extract financial data.

    Returns:
        None.
    """

    financials = \
        client.company_basic_financials(ticker, 'all')

    try:
        quarterly_data = financials['series']['quarterly']
    except:
        quarterly_data = {}
        print(f"From {__file__} FinancialsData extract_financials_data(): Ticker {ticker} does not have quarterly financial data.")
    finally:
        print(f"Extracting raw {ticker} financials data...")
        financial_df = pd.DataFrame(columns = ['period'])
        for financial, values in quarterly_data.items():
            financials_df = pd.DataFrame(values, columns = ['period', 'v'])
            financials_df.rename(columns = {"v": financial}, inplace = True)
            if financials_df.empty:
                financials_df = financial_df
            else:
                financials_df = financials_df.merge(financial_df, on = 'period', how = 'outer')

                csv_buf = StringIO()
                financials_df.to_csv(csv_buf, index = False)
                csv_buf.seek(0)

                s3.put_object(Bucket = AWS_S3_BUCKET, 
                              Body = csv_buf.getvalue(), 
                              Key = f'financials/{stock}.csv')
        print(f"Done extracting raw {ticker} financials data...")


if __name__ == "__main__":
    # load config
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

        extract_and_store_raw_financials_data(client = finnhub.Client(api_key = FINNHUB_API_KEY),
                                              ticker = stock)