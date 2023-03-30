import pandas as pd
import mlflow


class MovingAverageStrategy(mlflow.pyfunc.PythonModel):

    """
    A Python model for generating trading signals based on moving averages of a stock's price.

    Parameters:
        short_term_ma: The number of periods to use for the short-term moving average.
        long_term_ma: The number of periods to use for the long-term moving average.

    Methods:
        set_params(short_term_ma: int, long_term_ma: int) -> None:
            Sets the parameters for the moving average strategy.
        
        predict(context, data_points: pd.DataFrame) -> int:
            Generates trading signals based on the moving averages of a stock's price.
    """

    def __init__(self, short_term_ma: int = 0, long_term_ma: int = 200) -> None:

        """
        Initializes a MovingAverageStrategy object.

        Parameters:
            short_term_ma: The number of periods to use for the short-term moving average. Defaults to 0.
            long_term_ma: The number of periods to use for the long-term moving average. Defaults to 200.

        Returns:
            None.
        """

        self.short_term_ma = short_term_ma
        self.long_term_ma = long_term_ma

    def get_model_type(self):

        """
        Returns the model type
        """
        
        return "moving_average"

    def set_params(self, short_term_ma: int, long_term_ma: int) -> None:

        """
        Sets the parameters for the moving average strategy.

        Args:
            short_term_ma: The number of periods to use for the short-term moving average.
            long_term_ma: The number of periods to use for the long-term moving average.

        Returns:
            None.
        """

        self.short_term_ma = short_term_ma
        self.long_term_ma = long_term_ma

    def predict(self, context, data_points: pd.DataFrame) -> int:

        """
        Generates buy and sell signals based on the moving averages of the input data points.

        Args:
            context: The context object.
            data_points: A DataFrame of input data points with generated features.

        Returns:
            A Series of integer values representing buy (1) and sell (0) signals for each input data point.
        """

        if f"ma_t_{self.short_term_ma}" not in data_points.columns:
            raise ValueError(f"ma_t_{self.short_term_ma} not in generated features.")

        if f"ma_t_{self.long_term_ma}" not in data_points.columns:
            raise ValueError(f"ma_t_{self.long_term_ma} not in generated features.")

        def _get_signal(row):
            if row[f"ma_t_{self.short_term_ma}"] > row[f"ma_t_{self.long_term_ma}"]:
                return 1
            else:
                return 0
        
        return data_points.apply(lambda row: _get_signal(row), axis = 1)
