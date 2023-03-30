from persist import persist
from utilities import load_data, get_dates, load_mlflow_runs

import streamlit as st
import pandas as pd
import requests
import os
import yaml


with open(os.path.dirname(__file__) + '/config.yaml', 'r') as file:
	config = yaml.safe_load(file)

MLFLOW_SERVER = config["MLFLOW_SERVER"]
API_SERVER = config["API_SERVER"]

def get_universal_parameters(ticker: str, dates: pd.Series) -> dict:

    """
    Collects universal training parameters for a stock price prediction model.

    Args:
        ticker: The stock ticker symbol to train the model on.
        dates: A pandas series containing dates to use in the model training.

    Returns:
        dict: A dictionary containing the universal training parameters for the model.
    """

    name = st.text_input("What would you like to name your experiment?")

    target = \
        st.selectbox("What would you like to predict?", 
                        ('direction', 'volatility_t_5'),
                        key = persist("last_target"))

    horizon = \
        st.number_input('Over how many days would you like to validate during expanding window validation \
            (how large of a validation set would you like to have)?', 
            value = 30,
            key = persist("last_horizon_val"))
    window = \
        st.number_input('How many of the lastest days would you like to apply expanding window validation?', 
                        value = 100,
                        key = persist("last_window_val"))

    max_evals = \
        st.number_input('How many steps required to find optimal hyperparameter space?', 
                        value = 2,
                        key = persist("last_max_evals_val"))
    scores = \
        st.multiselect('Scores to record:', 
                        st.session_state.scores_to_record, 
                        key = persist("last_scores_to_record"))
    parallelism = \
        st.number_input('Number of hyper-parameter trials to run at once:', 
                        value = 2,
                        key = persist("last_parallelism_val"))

    if target == "direction":
        model_class = st.selectbox("What model class would you like to train?", 
                        ('ML', 'algo'),
                        key = persist("model_class_choice"))

        if model_class == "ML":
            model_type = st.selectbox('Select model type:', ('xgboost', 'test'))
        else:
            model_type = st.selectbox('Select model type:', ('moving_average', 'test'))

    elif target == "volatility_t_5":
        model_class = "ML"
        model_type = st.selectbox('Select model type:', ('xgboost', 'test'))

    # Collect universal training parameters
    experiment_parameters = {
        "ticker": ticker,
        "target": target,
        "name": name,
        "current_date": dates.iloc[st.session_state.start_date_index - st.session_state.day_number],
        "horizon" : int(horizon),
        "window" : int(window),
        "max_evals" : int(max_evals),
        "scores" : scores,
        "parallelism" : int(parallelism),
        'model_type': model_type,
        'model_class': model_class,
    }

    return experiment_parameters

def get_model_parameters(experiment_parameters: dict, stock_data: pd.DataFrame) -> None:

    """
    Collects model-specific training parameters based on experiment parameters and stock data.

    Args:
        experiment_parameters: A dictionary containing experiment parameters.
        stock_data: A pandas DataFrame containing stock data.

    Returns:
        None
    """

    # Collect model specific training parameters
    if experiment_parameters["model_type"] == 'xgboost':

        features = \
            st.multiselect('Features to use:', 
                            [c for c in stock_data.columns if c != experiment_parameters["target"] and c != 't'],
                            key = persist("features"))

        param_col_11, param_col_12, param_col_13, param_col_14 = st.columns(4)

        # depth
        step_size_depth = \
            param_col_11.number_input('max_depth step size:', value = 1)
        min_val_depth = \
            param_col_11.number_input('min depth:', value = 1)
        max_val_depth = \
            param_col_11.number_input('max depth:', value = 5)

        # n_estimators
        step_size_n_estimators = \
            param_col_12.number_input('step size n_estimators:', value = 1)
        min_val_n_estimators = \
            param_col_12.number_input('min n_estimators:', value = 150)

        max_val_n_estimators = \
            param_col_12.number_input('max n_estimators:', value = 200)

        # reg_alpha
        min_val_reg_alpha = \
            param_col_13.number_input('min reg_alpha:', value = 70)
        max_val_reg_alpha = \
            param_col_13.number_input('max reg_alpha:', value = 100)

        # max_delta_step
        min_val_max_delta_step = \
            param_col_14.number_input('min max_delta_step:', value = 3.5)
        max_val_max_delta_step = \
            param_col_14.number_input('max max_delta_step:', value = 4)

        st.markdown('''---''')

        param_col_21, param_col_22, param_col_23, param_col_24 = st.columns(4)

        # min_split_loss
        min_val_min_split_loss = \
            param_col_21.number_input('min min_split_loss:', value = 5)
        max_val_min_split_loss = \
            param_col_21.number_input('max min_split_loss:', value = 6)

        # learning_rate
        min_val_learning_rate = \
            param_col_22.number_input('min learning_rate:', value = 0.01)
        max_val_learning_rate = \
            param_col_22.number_input('max learning_rate:', value = 0.02)

        # min_child_weight
        min_val_min_child_weight = \
            param_col_23.number_input('min min_child_weight:', value = 1)
        max_val_min_child_weight = \
            param_col_23.number_input('max min_child_weight:', value = 3)

        # scale_pos_weight
        min_val_scale_pos_weight = \
            param_col_24.number_input('min scale_pos_weight:', value = 0.9)
        max_val_scale_pos_weight = \
            param_col_24.number_input('max scale_pos_weight:', value = 1.1)

        experiment_parameters.update({
            "features": features,
            'max_depth': {
                'min': int(min_val_depth),
                'max': int(max_val_depth),
                'step': int(step_size_depth)
                },

            'n_estimators': {
                'min': int(min_val_n_estimators),
                'max': int(max_val_n_estimators),
                'step': int(step_size_n_estimators)
            },
                
            'reg_alpha': {
                'min': min_val_reg_alpha,
                'max': max_val_reg_alpha
            },

            'max_delta_step': {
                'min': min_val_max_delta_step,
                'max': max_val_max_delta_step
            },
    
            'min_split_loss': {
                'min': min_val_min_split_loss,
                'max': max_val_min_split_loss
            },

            'learning_rate': {
                'min': min_val_learning_rate,
                'max': max_val_learning_rate
            },
                
            'min_child_weight': {
                'min': min_val_min_child_weight,
                'max': max_val_min_child_weight
            },
                
            'scale_pos_weight': {
                'min': min_val_scale_pos_weight,
                'max': max_val_scale_pos_weight
            }
        })


    elif experiment_parameters["model_type"] == "moving_average":
        # determine optimal short/long term moving average 
        min_short_term_ma, max_short_term_ma = \
            st.select_slider('Select range of short term ma to test on:', 
            options = list(range(5, 101, 5)),
            value = (5, 100))

        min_long_term_ma, max_long_term_ma = \
            st.select_slider('Select range of long term ma to test on:', 
            options = list(range(max_short_term_ma + 5, 201, 5)),
            value = (max_short_term_ma + 5, 200))

        experiment_parameters.update({
            "features": [c for c in stock_data.columns if 'ma_t_' in c and c != experiment_parameters["target"]] + ['returns'],
            'short_term_ma': {
                'min': int(min_short_term_ma),
                'max': int(max_short_term_ma),
                'step': int(5)
                },

            'long_term_ma': {
                'min': int(min_long_term_ma),
                'max': int(max_long_term_ma),
                'step': int(5)
            }
        })

def page_model(stock_data: pd.DataFrame, dates: pd.DataFrame) -> None:

    """
    This page allows you to train a model and deploy it to make predictions.
    The page will look for models for each stock type that was trained BEFORE the start date. (WIP)
    If there is a model, user can deploy it to make predictions, otherwise they can train a model
    """

    # Load data
    stock_data = load_data('SPY')
    dates = get_dates(data = stock_data)
    runs = load_mlflow_runs()

    # Option to train model on stock of choice
    ticker = st.selectbox(
        "Select stock data to train model on:",
        st.session_state.stock_list,
        key = persist("current_stock")
    )

    # Get date of latest trained model for particular stock
    try:
        st.metric(label = f"Latest Model for {ticker}: ", 
                  value = str(pd.to_datetime(runs.loc[~runs['tags.best_run'].isna()]['end_time'].iloc[0])))
    except:
        st.metric(label = f"Latest Model for {ticker}: ", 
                  value = "There are no trained models")

    st.write("Click button below to train model using the latest available data up to current date:")

    with st.expander("Set Model Experiment Parameters"):
        experiment_parameters = get_universal_parameters(ticker = ticker, 
                                                        dates = dates)

        # Collect model specific training parameters
        get_model_parameters(experiment_parameters = experiment_parameters, 
                            stock_data = stock_data)

        if st.button("TRAIN MODEL"):

            # call experiment run with stock
            uri = API_SERVER + \
                f"/experiment_{experiment_parameters['model_class']}_{experiment_parameters['model_type']}"

            data = requests.post(uri, json = experiment_parameters)
            print(data.text)

    # Table showing model training results
    with st.expander("Show experiment results"):
        st.write(f"Visit the mlflow UI via {MLFLOW_SERVER} to see more")
        try:
            st.dataframe(runs[runs['tags.mlflow.runName'].str.contains(ticker)], use_container_width = True)
        except:
            st.write("Train model to see experiment results")