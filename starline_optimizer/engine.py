from typing import ReadOnly
import numpy as np
import pandas as pd
import yfinance as yf
import cvxportfolio as cvx
from .data_provider import DataProvider

# u, t, shares_traded
type TradeResult = tuple[pd.Series, pd.Timestamp, pd.Series]


class OptimizationEngine:
    policies: ReadOnly[list[cvx.policies.Policy]]
    data: ReadOnly[DataProvider]

    def __init__(self, assets: list[str]):
        # TODO get asset information from clickhouse
        dataraw = yf.Tickers(assets).download()
        self.data = DataProvider(dataraw["Close"], dataraw["Volume"])
        self.policies = [
            self._make_policy(gr, gt)
            for gr in [5, 10, 20, 50, 100, 200, 500]
            for gt in [3, 3.5, 4, 4.5, 5, 5.5, 6]
        ]

    def _make_policy(self, gamma_risk: float, gamma_trade: float) -> cvx.policies.Policy:
        """Creates an optimization policy from the provided hyperparameters."""
        return cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast()
            - gamma_risk * cvx.FactorModelCovariance(num_factors=10)
            - gamma_trade * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],  # No shorting, no leverage
            planning_horizon=6,
            solver="ECOS",
        )

    def _cash_only(self) -> pd.Series:
        """Creates $1M cash only portfolios over the data's list of assets."""
        portfolio = pd.Series([0 for _ in self.data.tickers] + [1_000_000])
        portfolio.index = np.append(self.data.tickers, "USDOLLAR")
        return portfolio

    def execute(
        self, h: pd.Series, t: pd.Timestamp = None
    ) -> list[TradeResult]:
        """Executes all trading policies at current or user specified time.

        :param h: Holdings vector, in dollars, including the cash account (the last element).
        :param t: Time of execution
                  Defaults to earliest time in our data's trading calendar

        :return: List of trade weights, trade timestamps, and shares traded
        """
        if t is None:
            t = self.data.trading_calendar()[-1]
        return list(map(lambda p: p.execute(h, self.data, t), self.policies))
