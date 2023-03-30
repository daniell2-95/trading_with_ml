from persist import load_widget_state
from dashboard_page import page_dashboard
from model_training_page import page_model
from utilities import load_data, get_dates
from historical_data_page import page_historical_data
from diagnostics_page import page_diagnostics

import streamlit as st
import pandas as pd

PAGES = {
    "Dashboard": page_dashboard,
    "Historical Data": page_historical_data,
    "Train Model": page_model,
    "Diagnostics": page_diagnostics
}

def main() -> None:

    """
    Function that stores session states and executes the app pages
    """
    
    stock_data = load_data('SPY')
    dates = get_dates(data = stock_data)

    if "page" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "page": 0,
            "disable_date_input": False,
            "dates_set": set(dates),
            "lock_date": False,

            # REPLACE WITH CONFIG CONTAINING ALL STOCKS
            "stock_list": ["SPY", "TEST"],
            "current_stock": "SPY",
            "train_dates": {"SPY": []},

            # Dashboard
            "show_spy_checkbox": True,
            "show_spy": True,

            # Technical indicator multiselect options.
            "technical_indicators": [c for c in stock_data.columns if c != "direction"],
            "selected_indicators": [],

            # Default account values.
            "init_account_value": -1,
            "disable_inputs": False,
            "account": None,
            "start_date": pd.to_datetime(dates.iloc[500]),
            "start_date_index": 500,
            "account_returns_history": [],
            "order_history": [],
            "day_number": 0,
            "processed_days": set(),

            # Model values
            "model_deployment_history": [],
            "model_pred_matches": [1],
            "prediction_history": [],
            "prediction_correctness_history": [],
            "index_value_history": [0],
            "strategy_history": [],
            "features": ['returns', 'volatility_t_5', 'ma_t_5', 'ewma_t_5', 'momentum_t_5', 'stochastic_K', 'stochastic_D', 'rsi', 'macd', 'williams_r', 'a/d_oscillator', 'cci'],
            "model_class": None,

            # Model training page
            "current_model_name": None,
            "last_target": "direction",
            "model_class_choice": "ML",
            "last_horizon_val": 30,
            "last_window_val": 100,
            "last_max_evals_val": 2,
            "scores_to_record": ["accuracy", "roc_auc"],
            "last_scores_to_record": ["accuracy", "roc_auc"],
            "last_parallelism_val": 2,
            
        })

    # Page navigator
    page = st.sidebar.radio("Select your page", tuple(PAGES.keys()))
    if page == "Train Model":
        PAGES[page](stock_data = stock_data, dates = dates)
    else:
        PAGES[page]()


if __name__ == "__main__":
    load_widget_state()
    main()