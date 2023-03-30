import pandas as pd
import numpy as np

def get_returns(stock_data: pd.DataFrame) -> None:

    """
    Function that calculates 1 day returns using close and prev close price

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
    Returns:
        None. The function modifies the input DataFrame in place by adding a column: 'returns'
    """

    stock_data['returns'] = \
        np.log(stock_data['c'] / stock_data['c'].shift(1))
    
def get_prev_day_feature(stock_data: pd.DataFrame, column: str, lag = 1) -> None:

    """
    Function that calculates nth previous day features

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        column: str representing column name
        lag: int representing nth previous day

    Returns:
        None. The function modifies the input DataFrame in place by adding a column representing 
        the nth prev day feature with the name 'prev_{column_name}_{lag_number}'
    """

    if column in stock_data.columns:
        stock_data[f'prev_{column}_{lag}'] = \
            stock_data[column].shift(lag)
    else:
        raise ValueError

def get_volatility(stock_data: pd.DataFrame, window: int) -> None:

    """
    Function that calculates volatility using the previous n days

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        window: int representing prev n days to calculate volatility from

    Returns:
        None. The function modifies the input DataFrame in place by adding a column representing 
        volatility with the name 'volatility_t_{window_size}'
    """

    if 'returns' in stock_data.columns:
        stock_data[f'volatility_t_{window}'] = \
            stock_data['returns'].rolling(window).std()
    else:
        raise ValueError

def get_direction(stock_data: pd.DataFrame) -> None:

    """
    Function that calculates the direction (increase or decrease) based on current and prev close

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features

    Returns:
        None. The function modifies the input DataFrame in place by adding a column representing 
        the direction of stock price with the name 'volatility_t_{window_size}'
    """

    if 'returns' in stock_data.columns:

        def _apply_direction(val: float) -> int:
            if val > 0:
                return 1
            elif val <= 0:
                return 0

        stock_data['direction'] = \
            stock_data.apply(lambda row: _apply_direction(row['returns']), axis = 1)
    
    else:
        raise ValueError("returns does not exist in dataframe")
    
def get_moving_avg(stock_data: pd.DataFrame, window: int) -> None:

    """
    Function that calculates moving average using close prices over a window

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        window: int representing how many prev days to use to calculate moving average

    Returns:
        None. The function modifies the input DataFrame in place by adding a column representing 
        the moving average with the name 'ma_t_{window_size}'
    """

    stock_data[f'ma_t_{window}'] = \
        stock_data['c'].rolling(window).mean()

def get_exponential_weighted_moving_avg(stock_data: pd.DataFrame, window: int) -> None:

    """
    Function that calculates exponential moving average using close prices over a window

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        window: int representing how many prev days to use to calculate 
                exponential moving average

    Returns:
        None. The function modifies the input DataFrame in place by adding a column representing 
        the exponential moving average with the name 'ewma_t_{window_size}'
    """

    stock_data[f'ewma_t_{window}'] = \
        stock_data['c'].ewm(span = window).mean()

def get_momentum(stock_data: pd.DataFrame, window: int) -> None:

    """
    Function that calculates momentum using close prices over a window

    Parameters:
        stock_data: pandas dataframe containing OHLC price data and
                    various engineered features
        window: int representing how many prev days to use to calculate 
                momentum

    Returns:
        None. The function modifies the input DataFrame in place by adding a column representing 
        the momentum with the name 'momentum_t_{window_size}'
    """

    stock_data[f'momentum_t_{window}'] = \
        stock_data['c'].diff(periods = window)

def get_stochastic_oscillator(stock_data: pd.DataFrame, resistance_window = 14, rolling_window = 3) -> None:

    """
    Calculate the stochastic oscillator for a given stock using the high, low, 
    and close prices from a pandas DataFrame.

    Parameters:
        stock_data: A pandas DataFrame containing columns for 
                    the high, low, and close prices of a stock.
        resistance_window: The window size to use when calculating the highest 
                        and lowest prices over a given period. Default is 14.
        rolling_window: The window size to use when calculating the rolling 
                        average of the stochastic oscillator. Default is 3.

    Returns:
        None. The function modifies the input DataFrame in place by adding two columns for 
        the stochastic oscillator: 'stochastic_K' and 'stochastic_D'.
    """

    period_high = \
        stock_data['h'].rolling(resistance_window).max()

    period_low = \
        stock_data['l'].rolling(resistance_window).min()

    stock_data['stochastic_K'] = \
        (stock_data['c'] - period_low) * 100 / \
        (period_high - period_low)

    stock_data['stochastic_D'] = \
        stock_data['stochastic_K'].rolling(rolling_window).mean()

def get_relative_strength_index(stock_data: pd.DataFrame) -> None:

    """
    Calculate the Relative Strength Index (RSI) for a given stock using 
    the closing prices from a pandas DataFrame.

    Parameters:
        stock_data: A pandas DataFrame containing a column for the 
                    closing prices of a stock.

    Returns:
        None. The function modifies the input DataFrame in place by 
        adding a new column for the RSI, labeled 'rsi'.
    """

    diff = stock_data['c'].diff()
    up = diff.clip(lower = 0)
    down = -1 * diff.clip(upper = 0)
    ema_up = up.ewm(com = 13, adjust = False).mean()
    ema_down = down.ewm(com = 13, adjust = False).mean()
    stock_data['rsi'] = ema_up/ema_down

def get_macd(stock_data: pd.DataFrame) -> None:

    """
    Calculate the Moving Average Convergence Divergence (MACD) for a 
    given stock using the closing prices from a pandas DataFrame.

    Parameters:
        stock_data: A pandas DataFrame containing a column for the closing prices of a stock.

    Returns:
        None. The function modifies the input DataFrame in place by adding a new column 
        for the MACD, labeled 'macd'.
    """

    exp1 = stock_data['c'].ewm(span = 12, adjust=False).mean()
    exp2 = stock_data['c'].ewm(span = 26, adjust=False).mean()
    stock_data['macd'] = exp1 - exp2

def get_williams_r(stock_data: pd.DataFrame, window = 14) -> None:

    """
    Calculates and adds the Williams %R technical indicator to a DataFrame containing stock data.

    Args:
        stock_data: A DataFrame containing stock data with columns for high, low, and close prices.
        window: The number of periods to use when calculating the indicator. Defaults to 14.

    Returns:
        None. The Williams %R values are added to the input DataFrame as a new column named 'williams_r'.
    """

    period_high = \
        stock_data['h'].rolling(window).max()

    period_low = \
        stock_data['l'].rolling(window).min()
    
    close = stock_data['c']
    
    stock_data['williams_r'] = (period_high - close) / (period_high - period_low) * -100 

def get_ad_oscillator(stock_data: pd.DataFrame):

    """
    Calculate the Williams %R for a given stock using the high, low, and close prices 
    from a pandas DataFrame.

    Parameters:
        stock_data: A pandas DataFrame containing columns for the high, low, 
                    and close prices of a stock.
        window: The window size to use when calculating the highest and 
                lowest prices over a given period. Default is 14.

    Returns:
        None. The function modifies the input DataFrame in place by adding a new column 
        for the Williams %R, labeled 'williams_r'.
    """

    # Calculate AD oscillator feature
    high = stock_data['h']
    low = stock_data['l']
    close = stock_data['c']
    volume = stock_data['v']
    adl = ((close - low) - (high - close)) / (high - low) * volume
    stock_data['ad_oscillator'] = adl.rolling(window = 3).sum()

def get_cci(stock_data: pd.DataFrame, window = 20) -> None:

    """
    Calculate the Commodity Channel Index (CCI) for a given stock using the 
    high, low, and close prices from a pandas DataFrame.

    Parameters:
        stock_data: A pandas DataFrame containing columns for the high, low, 
                    and close prices of a stock.
        window: The window size to use when calculating the average and mean absolute 
                deviation over a given period. Default is 20.

    Returns:
        None. The function modifies the input DataFrame in place by adding a new column for the CCI, labeled 'cci'.
    """

    M = (stock_data['h'] + stock_data['l'] + stock_data['c']) / 3 
    SM = M.rolling(window).mean()
    D = M.rolling(window).apply(lambda x: pd.Series(x).mad())
    stock_data['cci'] = (M - SM) / (0.015 * D)