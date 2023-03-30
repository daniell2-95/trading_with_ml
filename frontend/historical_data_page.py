import streamlit as st
from utilities import load_data, get_dates
from dashboard_page import filter_df_by_dates, display_price_charts


def page_historical_data() -> None:
    """
    Page dedicated to showing historical data, price charts, and technical indicators
    so user can come up with strategies
    """
    # Load data
    spy_data = load_data('SPY')
    dates = get_dates(data = spy_data)

    current_date_index = st.session_state.start_date_index - st.session_state.day_number

    # Filter data by input date by user in dashboard
    filtered_spy_data = \
        filter_df_by_dates(stock_data = spy_data, 
                           start_date = dates.iloc[-1], 
                           end_date = dates.iloc[current_date_index])

    # Slider for user to further filter data
    start_date, end_date = st.select_slider(
            'Select a date range to filter historical data:',
            options = reversed(tuple(dates.iloc[current_date_index: ])),
            value = (dates.iloc[-1], dates.iloc[current_date_index]))
    
    st.subheader(f"Selected range: start date = :green[{start_date}], end_date = :red[{end_date}]")

    # Display price charts and technical indicators
    display_price_charts(stock_data = filtered_spy_data,
                         start_date = start_date,
                         end_date = end_date)

