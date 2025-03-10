import pandas as pd
import cvxportfolio as cvx

from .clickhouse import get_timespan
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
    """Serves market data for the optimization engine."""

    tickers: list[str]
    __prices: pd.DataFrame
    __return: pd.DataFrame
    __volume: pd.DataFrame

    def __init__(self, tickers: list[str]):
        """Initializes DataProvider with price and volume data.
        Both DataFrames must have pd.Timestamp indexes and columns with the price data.

        :param tickers: List of tickers for this DataProvider instance
        """
        for t in tickers:
            update_timeseries(f"series.{t}")
        data = list(map(lambda t: _tuples_to_df(get_timespan(f"series.{t}")), tickers))
        prices = map(lambda d, a: d[["price"]].rename({"price": a}, axis=1), data, tickers)
        volumes = map(lambda d, a: d[["volume"]].rename({"volume": a}, axis=1), data, tickers)

        prices_df = pd.concat(prices, axis=1)
        volumes_df = pd.concat(volumes, axis=1)

        # All database entries are type str, convert to actual useful values
        # If any price entry is missing from the dataframe use the previous date's entry
        prices_df = prices_df.ffill().map(float)
        volumes_df = volumes_df.fillna(0).map(int)

        self.tickers = tickers
        self.__prices = prices_df
        self.__return = prices_df.pct_change().fillna(0)
        self.__return["USDOLLAR"] = 0.04**252  # TODO temp risk-free rate value
        self.__volume = volumes_df  # TODO macro values have no volume

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
                raise pd.errors.DataError("Price data for DataProvider has duplicate timestamps.")
            if not include_end:
                end_date_pos -= 1
        else:
            end_date_pos = None

        return calendar[start_date_pos:end_date_pos]

    @property
    def periods_per_year(self) -> int:
        return 252  # TODO quarterly, monthly, etc

    @property
    def full_universe(self) -> pd.Index:
        """Full universe, which might not be available for trading.

        :return: Full universe.
        """
        return self.__prices.columns
