import json
import pandas as pd
import cvxportfolio as cvx
from typing import Callable

from .env import APP_ENV
from .logger import logger
from .clickhouse_timeseries import update_timeseries

# TODO Returns forecast dataframe from Clickhouse

# past_returns, current_returns, past_volumes, current_volumes, current_prices
type DataInstance = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]


def _tuples_to_df(data: list[tuple[pd.Timestamp, float, int]]) -> pd.DataFrame:
    """
    Converts a list of tuples that represent rows in the "series" database into a DataFrame

    :param data: List of database rows:
                 the first element is the timestamp
                 the second element is the price
                 the third element is the volume

    :return: A dataframe with timestamp index and price and volume columns
    """
    date, price, volume = zip(*data)
    date = pd.to_datetime(date)
    df = pd.DataFrame({"price": price, "volume": volume}, index=date)
    return df


class DataProvider(cvx.data.MarketData):
    """Serves market data for the optimization engine. """

    __id: str  # Jon id
    tickers: list[str]
    __prices: pd.DataFrame
    __return: pd.DataFrame
    __volume: pd.DataFrame

    def _default_prices_df(self, ch_table: str):
        """Generates a default prices DataFrame based on returns DataFrame and Clickhouse data.
        Uses the most recent Clickhouse asset prices as the current price of each ticker,
        then uses self.__return to generate all other rows.

        :return: DataFrame with same index, columns, shape as self.__return representing price data.
        """
        pass  # TODO implement

    def _default_volumes_df(self):
        """Generates a default volume DataFrame based on returns DataFrame.
        Sets volume to 1m for every period.
        Requires self.__return to exist before calling.

        :return: DataFrame with same index, columns, shape as self.__return,
                 but with all values set to 1_000_000
        """
        return self.__return.map(lambda x: 1_000_000)  # default to 1m shares traded every period

    def __init__(self, tickers: list[str], id: str, returns: pd.DataFrame):
        """Initializes DataProvider with price and volume data, which are required by cvxportfolio.

        :param tickers: List of tickers for this DataProvider instance
        """
        self.__id = id
        for t in tickers:
            update_timeseries(f"series.{t}")
        # TODO fetch the most recent entry from clickhouse
        # TODO generate prices df
        self.tickers = tickers
        self.__return = returns
        self.__prices = self.__return.map(lambda x: 50)
        self.__volume = self._default_volumes_df()
        self.__return["USDOLLAR"] = 1.04**(1/12)  # TODO temp risk-free rate value

        df_log = f"""{self.__id} DataProvider input data:\nReturns:\n{self.__return}
                     \nPrices:\n{self.__prices}
                     \nVolumes:\n{self.__volume}
                 """
        self._log(logger.debug, df_log)
        init_log = f"{self.__id} Successfully initalized DataProvider tickers {self.tickers}"
        self._log(logger.info, init_log)

    def _log(self, severity: Callable, message: str, addtl_fields: dict = None):
        """Logs a message.

        :param severity: One of logger.trace, logger.debug, logger.info, logger.success,
                         logger.warning, logger.error, logger.critical
        :param message: Log message
        :param addtl_fields: Additional JSON fields to log
        """
        if addtl_fields is None:
            addtl_fields = {}

        if APP_ENV == "production":
            severity(json.dumps({
                "job_id": self.__id,
                "tickers": self.tickers,
                "message": message,
                **addtl_fields
                }))
        else:
            if addtl_fields == {}:
                severity(message)
            else:
                severity(f"{message}\n{json.dumps(addtl_fields, indent=4)}")

    def serve(self, t: pd.Timestamp) -> DataInstance:
        """Serve data for policy and simulator at time t.

        :param t: Trading time. It must be included in the timestamps returned
                  by self.trading_calendar.

        :return: past_returns, current_returns, past_volumes, current_volumes, current_prices
        """
        date_pos = self.__prices.index.get_loc(t)

        if not isinstance(date_pos, int):
            raise pd.errors.DataError(f"Price data for DataProvider has duplicate timestamps {t}.")

        past_returns = self.__return.iloc[: date_pos - 1]
        curr_returns = self.__return.iloc[date_pos]
        past_volumes = self.__volume.iloc[: date_pos - 1]
        curr_volumes = self.__volume.iloc[date_pos]
        curr_prices = self.__prices.iloc[date_pos]

        self._log(logger.trace, f"{self.__id} Served data for time {t}")
        return (past_returns, curr_returns, past_volumes, curr_volumes, curr_prices)

    def trading_calendar(
        self,
        start_time: None | pd.Timestamp = None,
        end_time: None | pd.Timestamp = None,
        include_end: bool = True,
    ) -> pd.Index:
        """Get trading calendar between times.

        :param start_time: Initial time of the trading calendar. Always inclusive
                           if present. If None, use the first available time.
        :param end_time: Final time of the trading calendar. If None,
                         use the last available time.
        :param include_end: Include end time.

        :return: Trading calendar.
        """
        calendar = self.__prices.index
        start_date_pos = 0 if start_time is None else calendar.get_loc(start_time)

        if end_time is not None:
            end_date_pos = calendar.get_loc(end_time)
            if not isinstance(end_date_pos, int):
                self._log(logger.error, f"Price data for {self.__id} has duplicate timestamps.")
                raise pd.errors.DataError("Price data for DataProvider has duplicate timestamps.")
            if not include_end:
                end_date_pos -= 1
        else:
            end_date_pos = None

        return calendar[start_date_pos:end_date_pos]

    @property
    def periods_per_year(self) -> int:
        return 12  # TODO quarterly, monthly, etc

    @property
    def full_universe(self) -> pd.Index:
        """Full universe, which might not be available for trading.

        :return: Full universe.
        """
        return self.__prices.columns
