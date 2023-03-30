import os
import csv
import yaml
import boto3
import psycopg2

QUERIES = {
    'direction': "direction NOT IN (0, 1)",
    'stochastic_K': "stochastic_K < 0 OR stochastic_K > 100",
    'stochastic_D': "stochastic_D < 0 OR stochastic_D > 100",
    'rsi': "rsi < 0 OR rsi > 100",
    'macd': "macd > 1000",
    'williams_r': "williams_r > 0 OR williams_r < -100",
    'cci': "cci > 100 OR cci < -100"
}

def write_list_to_csv_file(feature_name: str, file_path: str, data: list) -> None:

    """
    Write a list of stock data to a CSV file, appending it to the end of
    the file if it already exists.

    Parameters:
        feature_name (str): The name of the feature being recorded.
        file_path (str): The path to the CSV file where the data should be written.
        data (list): A list of lists, where each inner list represents a single 
                     row of stock data in the order [open, high, low, close, volume, timestamp].

    Returns:
        None. The function writes the data to the specified CSV file.
    """

    # Create directory if it doesn't exist
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    if not os.path.isfile(file_path):
        # Create the file if it doesn't exist
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write the header row if the file doesn't exist
            writer.writerow([feature_name, 'o', 'h', 'l', 'c', 'v', 't'])
    
    # Append the data to the file
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def validate_staging_data(ticker: str, stg_table_name: str):

    """
    This function takes a ticker symbol and the name of a staging table as inputs. 
    It executes validation queries on the specified table for various features and 
    prints the validation results to the console. If a query fails, it also writes the failed rows to a CSV file.

    Parameters:
        ticker (str): The ticker symbol for the stock being validated
        stg_table_name (str): The name of the staging table to be validated

    Returns:
        None

    Note:
        This function relies on a QUERIES dictionary which contains the SQL queries for 
        validating each feature. These queries should be defined before calling this function.
    """

    # Execute validation queries
    for feature in QUERIES.keys():

        # Define query
        query = f"SELECT COUNT(*) FROM {stg_table_name} WHERE {QUERIES[feature]}"

        cur.execute(query)
        result = cur.fetchone()

        if feature == 'direction':
            # Only the first row should not have direction
            if result[0] > 1:

                cur.execute(f"SELECT {feature}, o, h, l, c, v, t FROM {stg_table_name} WHERE {QUERIES[feature]}")
                get_rows_result = cur.fetchall()

                file_path = f'{os.path.dirname(os.path.dirname(__file__))}/logs/validation_results/{ticker}/{feature}_results.csv'
                print(f"Printing failed rows for {query} to {file_path}")
                write_list_to_csv_file(feature_name = feature, file_path = file_path, data = get_rows_result)

            else:
                print(f"Data validation passed for query '{query}'")

        elif result[0] > 0:
           
            print(f"Data validation failed: {result[0]} records found for query '{query}'")

            cur.execute(f"SELECT {feature}, o, h, l, c, v, t FROM {stg_table_name} WHERE {QUERIES[feature]}")
            get_rows_result = cur.fetchall()

            file_path = f'{os.path.dirname(os.path.dirname(__file__))}/logs/validation_results/{ticker}/{feature}_results.csv'
            print(f"Printing failed rows for {query} to {file_path}")
            write_list_to_csv_file(feature_name = feature, file_path = file_path, data = get_rows_result)

        else:
            print(f"Data validation passed for query '{query}'")
            
def write_to_production(stg_table_name: str, prod_table_name: str) -> None:

    """
    Function that writes validated staging table to prod

    Parameters:
        stg_table_name: str representing staging table name
        prod_table_name: str representing prod table name

    Returns:
        None. Writes contents of staging table to prod
    """

    # Execute a CREATE TABLE AS statement to copy the data from the staging table to the production table
    create_table_sql = f'CREATE TABLE {prod_table_name} AS SELECT * FROM {stg_table_name}'
    cur.execute(create_table_sql)
    print(f"{prod_table_name} created successfully.")

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

    for stock in config['s_and_p_500'][:1]:
        validate_staging_data(ticker = stock, stg_table_name = f"{stock.lower()}_price_data_staging")
        write_to_production(stg_table_name = f"{stock.lower()}_price_data_staging", prod_table_name = f"{stock.lower()}_price_data_prod")


    # f"{stock}_price_data_raw"
    # Close the cursor and connection objects
    cur.close()
    conn.close()