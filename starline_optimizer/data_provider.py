from typing import ReadOnly
import pandas as pd
import cvxportfolio as cvx
import numpy as np

# TODO Returns forecast dataframe from Clickhouse

# past_returns, current_returns, past_volumes, current_volumes, current_prices
type DataInstance = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]


class DataProvider(cvx.data.MarketData):
    """Serves market data for the optimization engine."""

    tickers = ReadOnly[list[str]]
    __prices: ReadOnly[pd.DataFrame]
    __return: ReadOnly[pd.DataFrame]
    __volume: ReadOnly[pd.DataFrame]

    def __init__(self, prices: pd.DataFrame, volume: pd.DataFrame):
        """Initializes DataProvider with price and volume data.
        Both DataFrames must have pd.Timestamp indexes and columns with the price data.

        :param prices: Asset prices (or symbol values)
        :param volume: Trading volume
        """
        self.tickers = prices.columns.array
        self.__prices = prices
        self.__return = prices.pct_change().fillna(0)
        self.__return["USDOLLAR"] = 0.04  # TODO temp risk-free rate value
        self.__volume = volume  # TODO macro values have no volume, need to figure out a workaround

    def serve(self, t: pd.Timestamp) -> DataInstance:
        """Serve data for policy and simulator at time t.

        :param t: Trading time. It must be included in the timestamps returned
            by self.trading_calendar.

        :returns: past_returns, current_returns, past_volumes, current_volumes, current_prices
        """
        date_pos = self.__prices.index.get_loc(t)

        past_returns = self.__return.iloc[: date_pos - 1]
        curr_returns = self.__return.iloc[date_pos]
        past_volumes = self.__volume.iloc[: date_pos - 1]
        curr_volumes = self.__volume.iloc[date_pos]
        curr_prices = self.__prices.iloc[date_pos]

        return (past_returns, curr_returns, past_volumes, curr_volumes, curr_prices)

    def trading_calendar(
        self,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
        include_end: bool = True,
    ) -> pd.Series:
        """Get trading calendar between times.

        :param start_time: Initial time of the trading calendar. Always
            inclusive if present. If None, use the first available time.
        :param end_time: Final time of the trading calendar. If None,
            use the last available time.
        :param include_end: Include end time.

        :returns: Trading calendar.
        """
        calendar = self.__prices.index
        start_date_pos = 0 if start_time is None else calendar.get_loc(start_time)
        if end_time is not None:
            end_date_pos = calendar.get_loc(end_time) - (1 if not include_end else 0)
        else:
            end_date_pos = None
        return calendar[start_date_pos:end_date_pos]

    @property
    def periods_per_year(self) -> int:
        return 252  # TODO quarterly, monthly, etc

    @property
    def full_universe(self) -> pd.Index:
        """Full universe, which might not be available for trading.

        :returns: Full universe.
        """
        return self.__prices.columns

    @property
    def partial_universe_signature(self, partial_universe: pd.Index) -> str:
        """Unique signature of this instance with a partial universe.

        A partial universe is a subset of the full universe that is
        available at some time for trading.

        This is used in cvxportfolio.cache to sign back-test caches that
        are saved on disk. If not redefined it returns None which disables
        on-disk caching.

        :param partial_universe: A subset of the full universe.

        :returns: Signature.
        """
        assert np.all(partial_universe.isin(self.full_universe))
        result = f"{self.__class__.__name__}("
        result += f"datasource={self.datasource.__name__}, "
        result += f"partial_universe_hash={cvx.utils.hash_(np.array(partial_universe))},"
