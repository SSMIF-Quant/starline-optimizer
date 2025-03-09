from typing import Sequence
import numpy as np
import pandas as pd
import cvxportfolio as cvx
from .data_provider import DataProvider
from .threshold_constraints import ReturnsTarget

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
        self.t = self.data.trading_calendar()[-1]  # Current trading time
        self.risk_free_rate = 0.04
        self.policies = [
            self._make_policy(gr, gt)
            for gr in [5, 10, 20, 50, 100, 200, 500]
            for gt in [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5]
        ]

    def _default_r_forecast(self):
        """ Produces a new returns forecast instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.forecast.HistoricalMeanReturn()

    def _default_risk_metric(self):
        """ Produces a new risk metric instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.forecast.HistoricalFactorizedCovariance()

    def _make_policy(self, gamma_risk: float, gamma_trade: float) -> cvx.policies.Policy:
        """Creates an optimization policy from the provided hyperparameters."""
        rhat = self._default_r_forecast().estimate(self.data, self.t)
        return cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast(self._default_r_forecast())
            - gamma_risk * cvx.FullCovariance(self._default_risk_metric())
            - gamma_trade * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1), ReturnsTarget(rhat, 1.05)],
            # [cvx.LongOnly(), cvx.LeverageLimit(1)],
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

        :param h: Portfolio to calculate return for
                  If cash position is provided, uses the risk free rate for cash returns.

        :return: Expected return
        """
        w = h / np.sum(np.abs(h))  # Portfolio by asset weight
        exp_returns = self._default_r_forecast().estimate(self.data, self.t)
        # Include risk-free rate for cash position
        # exp_returns["USDOLLAR"] = self.risk_free_rate ** (1/252)
        return 1 + sum(map(lambda a, b: a * b, exp_returns, w))

    def h_risk(self, h: pd.Series) -> float:
        """Calculates expected risk for a portfolio produced by self.execute().

        :param h: Portfolio to calculate risk for
                  Must not contain the cash position

        :return: Expected risk
        """
        w = h / np.sum(np.abs(h))  # Portfolio by asset weight
        risk_mat = self._default_risk_metric().estimate(self.data, self.t)
        return w.T @ risk_mat @ w

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
