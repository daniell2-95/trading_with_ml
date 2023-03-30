#!/usr/bin/env python3

from airflow import DAG 
from airflow.operators.bash import BashOperator
import pendulum

main_path = "/opt/airflow/dags/"
schedule_interval = '@daily'
start_date = pendulum.now().subtract(days=1)

default_args = {"owner": "airflow", "depends_on_past": False, "retries": 1}

with DAG(
    dag_id = 'etl_stock_features',
    description ='Stock data ETL pipeline',
    schedule_interval = schedule_interval,
    default_args = default_args,
    start_date = start_date,
    catchup = True,
    max_active_runs = 1,
    tags = ['StockETL'],
) as dag:

    extract_raw_price_data = BashOperator(
        task_id = 'load_stock_price_data',
        bash_command = main_path + "/feature_generation/extract_raw_price_data.py",
        dag = dag,
    )
    extract_raw_price_data.doc_md = 'Extract stock price data and store as csv in S3 bucket.'

    extract_raw_financials_data = BashOperator(
        task_id = 'load_stock_financial_data',
        bash_command = main_path + "/feature_generation/extract_raw_financials_data.py",
        dag = dag,
    )
    extract_raw_financials_data.doc_md = 'Extract stock financial data and store as csv in S3 bucket.'

    validate_and_write_to_staging = BashOperator(
        task_id = 'validate_and_write_to_staging',
        bash_command = "feature_generation/validate_and_write_to_staging.py",
        dag = dag,
    )
    validate_and_write_to_staging.doc_md = 'Validate stock price data from S3 and write to Redshift staging'

    validate_and_write_to_prod = BashOperator(
        task_id = 'validate_and_write_to_prod',
        bash_command = "feature_generation/validate_and_write_to_prod.py",
        dag = dag,
    )
    validate_and_write_to_prod.doc_md = 'Validate generated features and write to Redshift prod.'

extract_raw_price_data >> validate_and_write_to_staging
extract_raw_financials_data >> validate_and_write_to_staging
validate_and_write_to_staging >> validate_and_write_to_prod