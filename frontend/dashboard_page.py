import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from account_info.account_class import Account
from plotly.subplots import make_subplots
from persist import persist
from utilities import load_data, get_dates, filter_df_by_dates, make_predictions


def get_model_prediction_correctness(stock_data: pd.DataFrame, 
                                     dates: pd.Series) -> None:
    """
    Function that modifies the session state containing list of model correctness.
    The function appends a value of 1 if the algo/ML strategy is correct for the
    current day, and 0 otherwise. Each element in the list represents correctness 
    for the day.

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        dates: pandas series containing dates in reverse order
    """

    # Get current day information
    curr_day = stock_data[stock_data['t'] == dates.iloc[st.session_state.start_date_index - \
    st.session_state.day_number]]

    # Append the current day return into the index value history
    st.session_state.index_value_history.append(curr_day['returns'].iloc[0])
    
    # Logic that appends 1/0 for correctness/incorrectness into prediction correctness
    # history for the current day. The prediction value 1 represents the expectation
    # that the price will go up, and 0 otherwise. We append 1 if the prediction from
    # the previous day is 1 and the current day returns are positive, otherwise 0
    if st.session_state.prediction_history[-1] == 1:
        if curr_day['returns'].iloc[0] > 0:
            st.session_state.model_deployment_history[-1]["prediction_correctness"].append(1)
        else:
            st.session_state.model_deployment_history[-1]["prediction_correctness"].append(0)
    else:
        if curr_day['returns'].iloc[0] < 0:
            st.session_state.model_deployment_history[-1]["prediction_correctness"].append(1)
        else:
            st.session_state.model_deployment_history[-1]["prediction_correctness"].append(0)

def get_current_day_predictions_and_positions(stock_data: pd.DataFrame, 
                                              dates: pd.Series) -> None:
    """
    Function that returns predictions for the next day given the current day as well
    as executing orders based on that prediction.

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        dates: pandas series containing dates in reverse order
    """

    # Get current day information
    #print("ACCOUNT CURRENT FUND ", st.session_state["account"].get_current_funds())
    curr_day = stock_data[stock_data['t'] == dates.iloc[st.session_state["start_date_index"] - \
                st.session_state["day_number"]]]
    
    prediction = make_predictions(model_name = st.session_state["current_model_name"], 
                                    data = curr_day[st.session_state["features"]])[0]

    # Store prediction history
    st.session_state["prediction_history"].append(prediction)

    # Execute orders based on predictions
    if prediction == 1:

        buy_status = st.session_state["account"].buy_shares(current_price = curr_day['c'].iloc[0], 
                                                            ticker = "SPY")
        st.session_state["order_history"].append(buy_status[1])

    else:

        sell_status = st.session_state["account"].sell_shares(current_price = curr_day['c'].iloc[0], 
                                                                ticker = "SPY")
        st.session_state["order_history"].append(sell_status[1])

    # Update account values to reflect current day changes
    st.session_state["account"].update_account_value(current_price = curr_day['c'].iloc[0])

    # Get latest account value history
    account_value_history = st.session_state["account"].get_account_value_history()

    # Update account returns
    st.session_state["account_returns_history"].append(account_value_history[-1] / account_value_history[-2])


def display_current_state_and_metrics(dates: pd.Series):

    """
    Function that displays latest metrics

    Parameters:
        dates: pandas series containing dates in reverse order
    """

    # Date information
    date_col1, date_col2, date_col3 = st.columns(3)
    date_col1.metric(label = "Start Date", 
                    value = str(st.session_state["start_date"].strftime('%Y-%m-%d')))
    date_col2.metric(label = "Current Date", 
                    value = dates.iloc[st.session_state["start_date_index"] - st.session_state["day_number"]])
    date_col3.metric(label = "Days Elapsed", 
                    value = st.session_state["day_number"])

    # Get relevent account information
    account_value_history = st.session_state["account"].get_account_value_history()
    account_fund_history = st.session_state["account"].get_funds_history()
    #print("ACCOUNT FUND HISTORY ", account_fund_history, "ACCOUNT_VALUE_HISTORY", account_value_history)

    # Display account metrics
    acc_col1, acc_col2, acc_col3 = st.columns(3)
    acc_col1.metric(label = "Current Day Gain", 
                    value = None \
                        if st.session_state["day_number"] == 0 \
                        else round(account_value_history[st.session_state["day_number"] + 1] - \
                            account_value_history[st.session_state["day_number"]], 2),

                    delta = '0' \
                        if st.session_state["day_number"] == 0 \
                        else str(round((account_value_history[st.session_state["day_number"] + 1] / \
                            account_value_history[st.session_state["day_number"]] - 1) * 100, 2)) + '%')

    acc_col2.metric(label = "Current Account Value", 
        value = 0 \
            if st.session_state["init_account_value"] == -1 or \
                st.session_state["current_model_name"] is None\
            else str(round(account_value_history[st.session_state["day_number"] + 1], 2)),
        delta = '0' \
            if st.session_state["day_number"] == 0 \
            else str(round((account_value_history[st.session_state["day_number"] + 1] / \
                st.session_state["init_account_value"] - 1) * 100, 2)) + '%')

    acc_col3.metric(label = "Remaining Funds", 
                    value = None \
                        if st.session_state["init_account_value"] == -1 or \
                            st.session_state["current_model_name"] is None\
                        else round(account_fund_history[st.session_state["day_number"] + 1], 2),
                    delta = '0' \
                        if st.session_state["init_account_value"] == -1 or \
                            st.session_state["current_model_name"] is None\
                        else str(round(account_fund_history[st.session_state["day_number"] + 1] - \
                            account_fund_history[st.session_state["day_number"]], 2)))


def display_returns(stock_data: pd.DataFrame, 
                    start_date: str, 
                    end_date: str) -> None:
    
    """
    Helper function to plot cumulative returns of the s&p500 as well as strategy returns

    Parameters:
        stock_data: stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        start_date: str representing start date
        end_date: str representing end date
    """

    st.header('Cumulative Returns')

    # Collect selected models in deployed history
    deployed_models = [model["model_name"] for model in st.session_state["model_deployment_history"]]
    model_list = st.multiselect("Select models to show past return performances",
                                deployed_models,
                                default = deployed_models)

    def _get_days_active() -> list:

        """
        Helper function used to get a list of days a model(s) has been active

        Parameters:
            None
        
        Returns:
            list containing the day numbers for each selected model representing when they were active
        """

        # List to collect the days that the selected models were active
        days_active = []
        for history in st.session_state["model_deployment_history"]:
            if history["model_name"] in model_list:
                # If a model has been selected by the user, get its start day
                start_day = history['day_deployed']

                # Check if current model history is a previously deployed model
                if st.session_state.day_number - start_day > len(history["prediction_correctness"]):

                    end_day = start_day + \
                        len(history['prediction_correctness'])

                # Otherwise, current model history is a currently deployed model
                else:
                    end_day = start_day + \
                        len(history['prediction_correctness'][: st.session_state.day_number - history["day_deployed"]])
                
                days_active += [day + 1 for day in range(start_day, end_day)]
        return days_active

    def _get_alpha_value() -> str:

        """
        Helper function used to calculate "alpha", which is the returns relative
        to the s&p500

        Parameters:
            None
        
        Returns:
            str representing the value of alpha
        """

        if st.session_state["init_account_value"] == -1:
            return None

        days_active = _get_days_active()
        val = round((np.prod([st.session_state["account_returns_history"][day] for day in days_active]) - \
            np.exp(np.sum([st.session_state["index_value_history"][day] for day in days_active]))) * 100, 2)

        if val > 0:
            return f":green[{val}%]"
        elif val < 0:
            return f":red[{val}%]"
        else:
            return "0%"

    def _get_accuracy() -> str:

        """
        Helper function used to calculate the accuracy of model predictions

        Parameters:
            None
        
        Returns:
            str representing the value of accuracy of model predictions
        """

        if st.session_state["init_account_value"] == -1:
            return None
        
        predictions = []
        for history in st.session_state["model_deployment_history"]:
            if history["model_name"] in model_list:
                if st.session_state.day_number - history['day_deployed'] > len(history["prediction_correctness"]):
                    predictions += history["prediction_correctness"]
                else:
                    predictions += history['prediction_correctness'][: st.session_state.day_number - history['day_deployed']]
        
        if not predictions:
            return None
        val = round(sum(predictions) / len(predictions) * 100, 2)
        if val > 0.5:
            return f":green[{val}%]"
        elif val < 0:
            return f":red[{val}%]"
        else:
            return "0%"

    def _get_days_deployed() -> int:

        """
        Helper function used to retrive the number of days a model has been deployed

        Parameters:
            None
        
        Returns:
            int representing the number of days a model has been deployed
        """

        days = 0
        for history in st.session_state["model_deployment_history"]:
            if history["model_name"] in model_list:
                if st.session_state.day_number - history['day_deployed'] <= len(history["prediction_correctness"]):
                    days += st.session_state.day_number - history['day_deployed']
                else:
                    days += len(history["prediction_correctness"])
        return days


    # Display alpha
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.subheader(f"Alpha: {_get_alpha_value()}")
    for history in reversed(st.session_state["model_deployment_history"]):
        if st.session_state.day_number >= history["day_deployed"]:
            metric_col2.subheader(f"Model Accuracy: {_get_accuracy()}")
            metric_col3.subheader(f"Days Deployed: {_get_days_deployed()}")
            break
        else:
            continue

    # Combine data so that it can be filtered by date
    correctness = [x for elem in st.session_state.model_deployment_history for x in elem["prediction_correctness"]]

    combined_df = pd.DataFrame({'t': stock_data['t'],
                                'day_number': list(range(st.session_state["day_number"] + 1)),
                                'cumulative_returns': np.exp(np.cumsum(st.session_state["index_value_history"][: st.session_state["day_number"] + 1])),
                                'model_predictions': [0] + correctness[: st.session_state["day_number"]],
                                'model_pred_returns': np.cumprod(st.session_state["account_returns_history"][: st.session_state["day_number"] + 1])})

    # Filter data
    filtered_stock_data = filter_df_by_dates(stock_data = combined_df,
                                             start_date = start_date,
                                             end_date = end_date)

    fig = go.Figure()

    # Option to display index returns on chart
    show_spy = st.checkbox("Show SPY?", key = persist("show_spy_checkbox"))
    if show_spy:
        fig.add_trace(go.Scatter(x = filtered_stock_data['t'], 
                                    y = filtered_stock_data['cumulative_returns'], 
                                    mode = "lines",
                                    name = "SPY Returns"))
        st.session_state["show_spy"] = True
    else:
        st.session_state["show_spy"] = False

    # Display current returns from strategy. The line will be colored green if the strategy was correct
    # for that day, and red otherwise.
    days_active = _get_days_active()
    def _get_line_color(stock_data, index) -> str:
        if stock_data['day_number'].iloc[index] in days_active:
            if stock_data['model_predictions'].iloc[index] == 1:
                return "green"
            elif stock_data['model_predictions'].iloc[index] == 0:
                return "red"
        else:
            return "gray"

    for i in range(1, filtered_stock_data.shape[0]):
        fig.add_trace(go.Scatter(x = filtered_stock_data['t'].iloc[i - 1: i + 1], 
                                 y = filtered_stock_data['model_pred_returns'].iloc[i - 1: i + 1],
                                 line_color = _get_line_color(filtered_stock_data, i),
                                 mode = "lines",
                                 showlegend = False))

    st.plotly_chart(fig)


def display_price_charts(stock_data: pd.DataFrame, 
                         start_date: str, 
                         end_date: str) -> None:
    """
    Helper function that plots price charts which include candlestick charts and 
    various technical indicators

    Parameters:
        stock_data: stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        start_date: str representing start date
        end_date: str representing end date
    """

    st.header('Price Chart')
    st.multiselect("Select technical indicator(s):", 
                    st.session_state["technical_indicators"], 
                    key = persist("selected_indicators"))

    # Filter data
    filtered_stock_data = filter_df_by_dates(stock_data = stock_data,
                                             start_date = start_date,
                                             end_date = end_date)

    # Initialize plot object
    fig = make_subplots(rows = 3, cols = 1, row_width = [0.2, 0.2, 0.6])

    # Candlestick charts
    fig.add_trace(go.Candlestick(x = filtered_stock_data['t'],
                                    open = filtered_stock_data['o'],
                                    high = filtered_stock_data['h'],
                                    low = filtered_stock_data['l'],
                                    close = filtered_stock_data['c']), row = 1, col = 1)
    fig.update(layout_xaxis_rangeslider_visible = False)

    # Volume chart
    fig.add_trace(go.Bar(x = filtered_stock_data['t'], 
                            y = filtered_stock_data['v'], 
                            name = 'Volume'), row = 3, col = 1)

    # Technical indicators
    for indicator in st.session_state["selected_indicators"]:
        if 'ma_t' in indicator or 'ewma_t' in indicator:
            fig.add_trace(go.Scatter(x = filtered_stock_data["t"], 
                                        y = filtered_stock_data[indicator], 
                                        name = indicator), row = 1, col = 1)
        else:
            fig.add_trace(go.Scatter(x = filtered_stock_data["t"], 
                                        y = filtered_stock_data[indicator], 
                                        name = indicator), row = 2, col = 1)

    # Configure plot size
    fig['layout'].update(height = 1000, width = 800)

    st.plotly_chart(fig)

def page_dashboard() -> None:
    """
    Function responsible for executing dashboard functions and displays
    """

    def _disable_inputs() -> None:
        """
        Internal helper function to disable inputs in dashboard
        """
        st.session_state["disable_inputs"] = True

    # Load data
    spy_data = load_data('SPY')
    dates = get_dates(data = spy_data)

    # Main title
    st.title('Trading With ML Simulator')

    # User input funds in account
    st.number_input('How much funds would you like to start with into your account?', 
                    key = persist("init_account_value"),
                    disabled = st.session_state.disable_inputs)

    # Initialize account class when user inputs funds
    if st.session_state.disable_inputs == True and \
        st.session_state.account is None and \
            st.session_state["init_account_value"] != -1:

        st.session_state.account = Account(st.session_state["init_account_value"])

    # User input start date to start simulation
    st.date_input('When would you like to start the simulation? \
        NOTE: It is recommended that you select the start date around 6 months before the end date.',
        value = pd.to_datetime(dates.iloc[500]),
        min_value = pd.to_datetime(dates.iloc[-1]),
        max_value =  pd.to_datetime(dates.iloc[0]),
        key = persist("start_date"),
        disabled = st.session_state.disable_inputs)

    # Lock inputs to start simulation
    st.checkbox("Confirm inputs (NOTE: This will lock the option to change initial account value and dates)", 
                key = persist("lock_date"),
                disabled = st.session_state.disable_inputs,
                on_change = _disable_inputs)

    # Delete all the items in Session state
    #if st.button('Restart Simulation?'):
    #    print(st.session_state)
    #    for key in st.session_state.keys():
    #        print(key)
    #        del st.session_state[key]
        
    # Get index of start date
    if str(st.session_state.start_date) in st.session_state.dates_set:
        st.session_state["start_date_index"] = \
            list(dates).index(str(st.session_state.start_date.strftime('%Y-%m-%d')))
    else:
        st.subheader(':red[Selected date is an invalid start date (weekend/holiday etc). Please select another date.]')

    # Buttons to move forward/backward a day
    if st.session_state["current_model_name"]:
        col1, col2 = st.columns(2)
        if col1.button('Get prev day'):

            # Prevent user from moving before start date
            if st.session_state.day_number == 0:
                st.subheader(':red[Cannot go back before the simulation start date.]')
            else:
                # Otherwise, decrement current day number
                assert len(st.session_state.index_value_history) > 0, \
                    "History of value changes should be greater than 0."

                st.session_state.day_number -= 1

        if col2.button('Get next day'):

            # Prevent user from going beyond available date
            if dates.iloc[st.session_state.start_date_index - st.session_state.day_number] == dates.iloc[0]:
                st.subheader(':green[Reached the latest available date.]')
            else:
                # Otherwise, increment day number and get model prediction correctness for the day
                st.session_state.day_number += 1

                # Condition is applied to only compute when entering a new day. Otherwise, past
                # data is stored to avoid unnecessary recomputation
                if st.session_state["day_number"] not in st.session_state["processed_days"]:
                    get_model_prediction_correctness(stock_data = spy_data, dates = dates)

    # Message indicating which model is deployed
    # TODO: GET MODEL SPECIFIC ACCURACY
    if st.session_state.current_model_name is None:
        st.subheader(f"There is no algorithm/ML strategy to make predictions. Please refer to the historical data and model training page to experiment on a strategy.")
    else:
        for history in reversed(st.session_state["model_deployment_history"]):
            if st.session_state.day_number >= history["day_deployed"]:
                st.subheader(f"Currently deployed strategy: :blue[{history['model_name']}]")
                break
            else:
                continue


    # Below operations are only possible if a model is deployed
    if st.session_state.current_model_name is not None:
        
        # Condition applied to compute only when necessary
        if st.session_state["day_number"] not in st.session_state["processed_days"]:

            # Function to get predictions for next day and positions
            get_current_day_predictions_and_positions(stock_data = spy_data, 
                                                      dates = dates)
                    
        # Store days that were processed to avoid recomputation
        st.session_state["processed_days"].add(st.session_state["day_number"])

        # Message to show user what the prediction for the next day is and what position
        # the algorithm/model requested
        st.subheader(st.session_state["order_history"][st.session_state["day_number"]])
        #print("MODEL CORRECTNESS ", st.session_state.prediction_correctness_history)
        display_current_state_and_metrics(dates = dates)

                    

    # Filter data by start date inputed by user and current day
    filtered_spy_data = \
        filter_df_by_dates(stock_data = spy_data, 
                           start_date = str(st.session_state.start_date.strftime('%Y-%m-%d')), 
                           end_date = dates.iloc[st.session_state.start_date_index - \
                            st.session_state.day_number])

    # Slider to filter simulation data by dates
    if st.session_state.day_number > 0:
        
        start_index = st.session_state.start_date_index - st.session_state.day_number
        end_index = st.session_state.start_date_index + 1

        start_date, end_date = st.select_slider(
            'Select a date range to filter the plots',
            options = reversed(tuple(dates.iloc[start_index: end_index])),
            value = (str(st.session_state.start_date.strftime('%Y-%m-%d')), 
                     dates.iloc[st.session_state.start_date_index - st.session_state.day_number]))
    else:
        start_date = end_date = st.session_state.start_date.strftime('%Y-%m-%d')

    # Display current model metrics and price charts
    if st.session_state.current_model_name is not None:
        display_returns(stock_data = filtered_spy_data,
                        start_date = start_date,
                        end_date = end_date)
        display_price_charts(stock_data = filtered_spy_data,
                             start_date = start_date,
                             end_date = end_date)