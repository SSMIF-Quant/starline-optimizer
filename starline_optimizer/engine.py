from typing import ReadOnly
import pandas as pd
import yfinance as yf
import cvxportfolio as cvx
import numpy as np

# TODO Returns forecast dataframe from Clickhouse

# past_returns, current_returns, past_volumes, current_volumes, current_prices
type DataInstance = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]


class DataProvider(cvx.data.MarketData):
    """Serves market data for the optimization engine. """
    tickers = ReadOnly[list[str]]
    __prices: ReadOnly[pd.DataFrame]
    __return: ReadOnly[pd.DataFrame]
    __volume: ReadOnly[pd.DataFrame]

    def __init__(self, assets: list[str]):
        # TODO append usdollar info to __return
        # TODO generalize prices, return, volume
        self.assets = assets
        data = yf.Tickers(assets).download().bfill()
        data.index = pd.to_datetime(data.index)
        self.__prices = data["Close"]
        self.__return = self.__prices.pct_change().fillna(0)
        self.__volume = data["Volume"]

    def serve(self, t: pd.Timestamp) -> DataInstance:
        """Serve data for policy and simulator at time t.

        :param t: Trading time. It must be included in the timestamps returned
            by self.trading_calendar.

        :returns: past_returns, current_returns, past_volumes, current_volumes, current_prices
        """
        date_pos = self.__prices.index.get_loc(t)

        past_returns = self.__return.iloc[:date_pos-1]
        curr_returns = self.__return.iloc[date_pos]
        past_volumes = self.__volume.iloc[:date_pos-1]
        curr_volumes = self.__volume.iloc[date_pos]
        curr_prices = self.__prices.iloc[date_pos]

        return (past_returns, curr_returns, past_volumes, curr_volumes, curr_prices)

    def trading_calendar(self, start_time: pd.Timestamp = None, end_time: pd.Timestamp = None,
                         include_end: bool = True) -> pd.Series:
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
        return 252

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
        result = f'{self.__class__.__name__}('
        result += f'datasource={self.datasource.__name__}, '
        result += f'partial_universe_hash={cvx.utils.hash_(np.array(partial_universe))},'


class OptimizationEngine:
    policies: ReadOnly[list[cvx.policies.Policy]]
    __data: ReadOnly[DataProvider]

    def __init__(self, assets: list[str]):
        # TODO get returns/risk metrics from clickhouse
        self.__data = DataProvider(assets)
        self.policies = [
                self._make_policy(gr, gt)
                for gr in [2, 5, 10, 20, 50, 100, 200, 500]
                for gt in [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
                ]

    def _make_policy(self, gamma_risk: float, gamma_trade: float) -> cvx.policies.Policy:
        """Creates an optimization policy from the provided hyperparameters. """
        return cvx.MultiPeriodOptimization(
                cvx.ReturnsForecast()
                - gamma_risk * cvx.FactorModelCovariance(num_factors=10)
                - gamma_trade * cvx.StocksTransactionCost(),
                [cvx.LongOnly(), cvx.LeverageLimit(1)],  # No shorting, no leverage
                planning_horizon=6, solver='ECOS'
            )

    def _cash_only(self) -> pd.Series:
        """Creates $1M cash only portfolios over the supplied list of assets. """
        return pd.Series([0 for _ in self.__data.tickers] + [1_000_000])

    def execute(self):
        cash_p = self._cash_only()
        yesterday = pd.Timestamp.date(pd.Timestamp.now()) - pd.Timedelta(1, "day")
        return list(map(lambda p: p.execute(cash_p, None, yesterday),  self.policies))
