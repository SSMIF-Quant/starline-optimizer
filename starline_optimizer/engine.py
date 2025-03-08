from typing import Sequence
import numpy as np
import pandas as pd
import cvxportfolio as cvx
from .data_provider import DataProvider

# u, t, shares_traded
type TradeResult = tuple[pd.Series, pd.Timestamp, pd.Series]

# expected return, expected risk
type PortfolioPerformance = tuple[float, float]


class RiskThresholdConstraint(cvx.constraints.InequalityConstraint):
    pass


class OptimizationEngine:
    policies: list[cvx.policies.Policy]
    data: DataProvider
    t: pd.Timestamp
    risk_free_rate: float

    def __init__(self, assets: list[str]):
        # TODO get return forecasts from clickhouse and pass into r_forecast
        self.data = DataProvider(assets)
        self.t = self.data.trading_calendar()[-1]
        self.risk_free_rate = 0.04
        self.policies = [
            self._make_policy(gr, gt)
            for gr in [5, 10, 20, 50, 100, 200, 500]
            for gt in [3, 3.5, 4, 4.5, 5, 5.5, 6]
        ]

    def _default_r_forecast(self):
        """ Produces a new returns forecast instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.forecast.HistoricalMeanReturn()

    def risk_metric(self):
        """ Produces a new risk metric instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.FactorModelCovariance(num_factors=10)

    def _make_policy(self, gamma_risk: float, gamma_trade: float) -> cvx.policies.Policy:
        """Creates an optimization policy from the provided hyperparameters."""
        return cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast(self._default_r_forecast())
            - gamma_risk * self.risk_metric()
            - gamma_trade * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            planning_horizon=6,
            solver="ECOS",
        )

    def _cash_only(self) -> pd.Series:
        """Creates $1M cash only portfolios over the data's list of assets."""
        portfolio = pd.Series([0 for _ in self.data.tickers] + [1_000_000])
        portfolio.index = np.append(self.data.tickers, "USDOLLAR")
        return portfolio

    def h_return(self, h: pd.Series) -> float:
        """Calculates expected return for a portfolio produced by self.execute().

        :param h: Portfolio to calculate return for, in USDOLLARS

        :return: Expected return
        """
        exp_returns = self._default_r_forecast().estimate(self.data, self.t)
        return 1 + sum(map(lambda a, b: a * b, exp_returns, h))

    def execute(self, h: pd.Series, t: pd.Timestamp = None) -> Sequence[TradeResult]:
        """Executes all trading policies at current or user specified time.

        :param h: Holdings vector, in dollars, including the cash account (the last element).
        :param t: Time of execution
                  Defaults to earliest time in our data's trading calendar

        :return: List of trade weights, trade timestamps, and shares traded
        """
        if t is None:
            t = self.t
        return list(map(lambda p: p.execute(h, self.data, t), self.policies))
        # portfolios = map(lambda t: t[0] + w, trades)
        # return map(lambda p: (self._portfolio_return(p[0][:-1]), p), portfolios)
