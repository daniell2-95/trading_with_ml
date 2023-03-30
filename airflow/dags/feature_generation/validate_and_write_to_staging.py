import boto3
import os
import yaml
import pandas as pd
import psycopg2
import json
from psycopg2.extras import execute_values
from datetime import date
from technical_indicators import (get_ad_oscillator, get_cci, get_direction, 
                                  get_exponential_weighted_moving_avg, get_macd, 
                                  get_momentum, get_moving_avg, get_prev_day_feature, 
                                  get_relative_strength_index, get_returns, 
                                  get_stochastic_oscillator, get_volatility, get_williams_r)


def load_from_s3(ticker: str) -> dict:

    """
    Load price data for a given stock ticker from an S3 bucket.

    Parameters:
        ticker: The stock ticker for which to retrieve the price data.

    Returns:
        dict: A dictionary representing the price data for the given stock ticker.
    """

    response = s3.get_object(Bucket = AWS_S3_BUCKET, 
                             Key = f"prices/{ticker}.json")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        try:
            data = json.loads(response.get("Body").read())
            return data
        except:
            raise ValueError(f"From {os.path.dirname(__file__)}/validate_price_data.py: Unsuccessful read of json file")
    else:
        raise ValueError(f"From {os.path.dirname(__file__)}/validate_price_data.py: Unsuccessful S3 get_object response. Status - {status}")

def validate_price_data(stock_json: dict) -> None:
    
    """
    Validate the given stock price data in JSON format.

    Parameters:
        stock_json: A dictionary containing the stock price data in JSON format. 
                    The dictionary should have the following keys: "o", "h", "l", "c", "v", "t", and "s".

    Returns:
        None. If the data passes all the checks, a success message is printed.
    """

    # Validate the JSON data
    # Here we assume that the data has the following fields: "o", "h", "l", "c", "v", "t", "s"
    if any(key not in stock_json.keys() for key in ["o", "h", "l", "c", "v", "t", "s"]):
        raise ValueError("JSON data is missing required fields")
    
    # Check if all values in JSON are same length
    length = len(next(iter(stock_json.values())))
    if any(len(stock_json[col]) != length for col in stock_json.keys() if col != 's'):
        raise ValueError("JSON data has uneven lengths")

    # Check if any values in JSON are below 0
    if any(min(stock_json[key]) < 0 for key in ["o", "h", "l", "c", "v"]):
        raise ValueError("OHLCV data contains values less than 0")
    
    # Check if the latest date is not later than the current date
    TODAY = date.today()
    if pd.to_datetime(max(stock_json["t"]), unit = 's', origin = 'unix') > TODAY:
        raise ValueError("The max date in JSON is greater than today's date")
    
    print(f"Successfully validated raw price data.")


def generate_features(stock_json: dict) -> pd.DataFrame:

    """
    Generates features for stock data.

    Args:
        stock_json: A dictionary containing stock data.

    Returns:
        pd.DataFrame: A Pandas dataframe containing the generated features.
    """

    stock_data = pd.DataFrame.from_dict(stock_json)

    # Transform time column into datetime format
    stock_data['t'] = pd.to_datetime(stock_data['t'], unit = 's', origin = 'unix')

    # Generate returns and volatility
    get_returns(stock_data = stock_data)
    get_direction(stock_data = stock_data)
    get_volatility(stock_data = stock_data, window = 5)

    # Generate lags for returns and volatility as features
    for i in range(1, 11):
        get_prev_day_feature(stock_data = stock_data, column = 'returns', lag = i)
        get_prev_day_feature(stock_data = stock_data, column = 'volatility_t_5', lag = i)

    # Generate Technical indicators
    for i in range(5, 201, 5):
        get_moving_avg(stock_data = stock_data, window = i)
        get_exponential_weighted_moving_avg(stock_data = stock_data, window = i)
        
    get_momentum(stock_data = stock_data, window = 5)
    get_stochastic_oscillator(stock_data = stock_data)
    get_relative_strength_index(stock_data = stock_data)
    get_macd(stock_data = stock_data)
    get_williams_r(stock_data = stock_data)
    get_ad_oscillator(stock_data = stock_data)
    get_cci(stock_data = stock_data)

    print(f"Successfully generated features.")

    return stock_data

def write_to_staging(stock_data: pd.DataFrame, table_name: str) -> None:

    """
    This function writes stock data to a staging table in Redshift. It first checks whether 
    the table exists in Redshift or not. If the table already exists, it checks whether the 
    table is empty or not. If the table is empty, it inserts the entire dataframe into the table. 
    Otherwise, it compares the latest date in the table with the latest date in the dataframe. 
    If the latest date in the dataframe is newer than the latest date in the table, 
    it inserts the last row of the dataframe into the table. Otherwise, 
    it doesn't insert anything. If the table doesn't exist, it creates the 
    table using the schema from the dataframe and inserts the entire dataframe into the table.

    Parameters:
        stock_data: A pandas DataFrame containing the stock data to be written to the staging table.
        table_name: A string containing the name of the staging table to write the data to.

    Returns:
        None
    """

    # Check if the table already exists in Redshift
    cur.execute(f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'")
    table_exists = cur.fetchone()[0]

    if table_exists:

        # Check if the table is empty
        cur.execute(f"SELECT count(*) FROM {table_name}")
        table_is_empty = cur.fetchone()[0] == 0

        if table_is_empty:
            # Fill the table with the entire dataframe
            columns = list(stock_data.columns)
            sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES %s"
            execute_values(cur, sql, stock_data.values.tolist())
            conn.commit()
            print(f"{stock_data.shape[0]} rows written to {table_name} table successfully.")

        else:
            # Fetch the latest date in the table
            cur.execute(f"SELECT MAX(t) FROM {table_name}")
            latest_date = cur.fetchone()[0]

            # Get the latest date in the dataframe
            df_latest_date = stock_data['t'].max()
            if df_latest_date > latest_date:
                # Fill the table with the last row of the dataframe
                columns = list(stock_data.columns)
                sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES %s"
                execute_values(cur, sql, stock_data.iloc[[-1]].values.tolist())
                conn.commit()
                print(f"{stock_data.iloc[[-1]].shape[0]} rows written to {table_name} table successfully.")
            else:
                print(f"No new data to insert into {table_name} table.")

    else:
        # Create the table using the first row of the dataframe as the schema
        create_table_query = f"CREATE TABLE {table_name} ({', '.join([f'{col.lower()} TEXT' if col in {'s', 't'} else f'{col.lower()} float8' for col in stock_data.columns])})"
        cur.execute(create_table_query)
        conn.commit()
        print(f"Table {table_name} created successfully.")

        # Fill the table with the entire dataframe
        columns = list(stock_data.columns)
        sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES %s"
        execute_values(cur, sql, stock_data.iloc[:-1].values.tolist())
        conn.commit()
        print(f"{stock_data.shape[0]} rows written to {table_name} table successfully.")


if __name__ == "__main__":

    with open(os.path.dirname(__file__) + '/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    AWS_S3_BUCKET = config['AWS_S3_BUCKET']
    AWS_ACCESS_KEY_ID = config['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = config['AWS_SECRET_ACCESS_KEY']

    # s3 client
    s3 = boto3.client('s3',
                      aws_access_key_id = AWS_ACCESS_KEY_ID, 
                      aws_secret_access_key = AWS_SECRET_ACCESS_KEY)

    # Connect to the Redshift cluster
    conn = psycopg2.connect(
        host = config["REDSHIFT_CREDS"]["HOST"],
        port = config["REDSHIFT_CREDS"]["PORT"],
        dbname = config["REDSHIFT_CREDS"]["DATABASE"],  
        user = config["REDSHIFT_CREDS"]["USER"],
        password = config["REDSHIFT_CREDS"]["PASSWORD"]
    )

    # Set up the connection and cursor objects
    cur = conn.cursor()

    for stock in config['s_and_p_500'][:11]:
        stock_json = load_from_s3(ticker = stock)
        validate_price_data(stock_json = stock_json)

        stock_data = generate_features(stock_json = stock_json)
        write_to_staging(stock_data = stock_data, table_name = f"{stock.lower()}_price_data_staging")

    # Close the cursor and connection objects
    cur.close()
    conn.close()