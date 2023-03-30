import finnhub
from collections import defaultdict
import os
from typing import List, Union

class Account:

    """
    Attributes:
        funds: the initial amount of funds in the account

    Methods:
        get_account_value_history(): returns the history of account values
        get_funds_history(): returns the history of account funds
        get_portfolio(): returns the current portfolio
        get_current_funds(): returns the current amount of funds in the account
        buy_shares(current_price: float, ticker: str) -> List[Union[bool, str]]: 
            executes a buy order for a specified ticker and price
        sell_shares(current_price: float, ticker: str) -> List[Union[bool, str]]: 
            executes a sell order for a specified ticker and price
        update_account_value(current_price: float) -> None: 
            updates the account value based on the closing price of all shares in the 
            portfolio and the current funds in the account
    """

    def __init__(self, funds: float):

        """
        Initializes an Account object with a given amount of funds.

        Parameters:
            funds: the amount of funds to be set for the account

        Returns:
            None
        """

        self.__current_funds = funds
        self.__portfolio = defaultdict(int)
        self.__account_history = {"account_values": [funds],
                                  "account_funds": [funds]}

    def get_account_value_history(self):

        """
        Returns a list of the historical values of the account.

        Parameters:
            None

        Returns:
            list: a list of the historical values of the account
        """

        return self.__account_history["account_values"]

    def get_funds_history(self):

        """
        Returns a list of the historical values of the funds in the account.

        Parameters:
            None

        Returns:
            list: a list of the historical values of the funds in the account
        """

        return self.__account_history["account_funds"]
    
    def get_portfolio(self):

        """
        Returns a dictionary of the stocks and the quantities owned in the account.

        Parameters:
            None

        Returns:
            dict: a dictionary of the stocks and the quantities owned in the account
        """

        return self.__portfolio

    def get_current_funds(self):

        """
        Returns the current amount of funds in the account.

        Parameters:
            None

        Returns:
            float: the current amount of funds in the account
        """

        return self.__current_funds
    
    def buy_shares(self, current_price: float, ticker: str) -> List[Union[bool, str]]:

        """
        Executes a buy order for the specified ticker and price if the current price is 
        less than or equal to the current funds in the account. Returns a list with a 
        boolean indicating if the order was executed and a string message with 
        details about the executed order or the reason why it was not executed.
 
        Args:
            current_price: The current price for the ticker
            ticker: The ticker for the stock to buy
            
        Returns: 
            A list with a boolean indicating if the order was executed and a string 
            message with details about the executed order or the reason why it was not executed.

        """

        #quote = client.quote(ticker)
        #rice = quote['c']
        if current_price > self.__current_funds:
            #print(f"From {os.path.basename(__file__)} Account buy_shares(): Order exceeds account funds, will not execute trade.")
            return [False, f"{ticker} prices are expected to rise tomorrow. Order exceeds account funds, will not execute trade."]

        else:
            quantity = self.__current_funds // current_price
            self.__portfolio[ticker] += quantity
            self.__current_funds -= current_price * quantity
            #print(f"From {os.path.basename(__file__)} Account buy_shares(): Executed order, bought {quantity} units of {ticker} for {current_price} each for a total of {quantity * current_price}.")
            return [True, f"{ticker} prices are expected to rise tomorrow. Executed order, bought {quantity} units of {ticker} for {current_price} each for a total of {round(quantity * current_price, 2)}."]
    
    def sell_shares(self, current_price: float, ticker: str) -> List[Union[bool, str]]:

        """
        Executes a sell order for a specified ticker and price.

        Parameters:
            current_price: The current price of the shares to be sold.
            ticker: The ticker symbol of the shares to be sold.

        Returns:
            A list containing a boolean indicating whether the sell order was executed successfully or not, 
            and a string describing the outcome of the order. If the order was unsuccessful, 
            the string will provide a reason for the failure.
        """

        # if ticker not in self.__portfolio:
        #     raise ValueError(f"From {os.path.basename(__file__)} Account sell_shares(): Attempted to sell shares not in portfolio.")

        #quote = client.quote(ticker)
        #price = quote['c']
        if self.__portfolio[ticker] == 0:
            #print(f"From {os.path.basename(__file__)} Account sell_shares(): Attempted to sell more shares than in account, will not execute trade.")
            return [False, f"{ticker} prices are expected to fall tomorrow. Attempted to sell more shares than in account, will not execute trade."]

        else:
            quantity = self.__portfolio[ticker]
            self.__current_funds += quantity * current_price
            self.__portfolio[ticker] = 0
            return [True, f"{ticker} prices are expected to fall tomorrow. Executed order, sold {quantity} units of {ticker} for {current_price} for a total of {round(quantity * current_price, 2)}."]

    def update_account_value(self, current_price: float) -> None:

        """
        Update the value of the account based on the current market price of the portfolio.

        Parameters:
            current_price: The current market price of the stocks in the portfolio.

        Returns:
            None. The function does not return anything but updates the instance variables of the class.
        """

        value = 0
        for stock in self.__portfolio:
            #quote = client.quote(stock)
            value += current_price * self.__portfolio[stock]

        self.__account_value = value + self.__current_funds
        self.__account_history["account_values"].append(self.__account_value)
        self.__account_history["account_funds"].append(self.__current_funds)
