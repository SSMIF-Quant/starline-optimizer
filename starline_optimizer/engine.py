import numpy as np
import pandas as pd
import cvxportfolio as cvx
from .data_provider import DataProvider
from .threshold_constraints import ReturnsTarget, RiskThreshold

# u, t, shares_traded
type TradeResult = tuple[pd.Series, pd.Timestamp, pd.Series]

# expected return, expected risk
type PortfolioPerformance = tuple[float, float]


class OptimizationEngine:
    data: DataProvider
    t: pd.Timestamp
    risk_free_rate: float

    def __init__(self, assets: list[str]):
        # TODO get return forecasts from clickhouse and pass into r_forecast
        self.data = DataProvider(assets)
        self.t = self.data.trading_calendar()[-1]  # Current trading time
        self.risk_free_rate = 1.04 ** (1/252)  # TODO temp non-annualized risk free rate

    def _default_r_forecast(self):
        """Produces a new returns forecaster instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.forecast.HistoricalMeanReturn()

    def _default_risk_metric(self):
        """Produces a new risk metric instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.forecast.HistoricalFactorizedCovariance()

    def _make_policy(self, gamma_risk: float, gamma_trade: float,
                     constraints: list[cvx.constraints.Constraint]) -> cvx.policies.Policy:
        """Creates an optimization policy from the provided hyperparameters.

        :param gamma_risk: Risk aversion hyperparameter
        :param gamma_trade: Trade aversion hyperparameter
        :param constraints: Extra constraints for the optimizer
        """
        return cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast(self._default_r_forecast())
            - gamma_risk * cvx.FullCovariance(self._default_risk_metric())
            - gamma_trade * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1), *constraints],
            planning_horizon=6,
            solver="ECOS",
        )

    def _cash_only(self) -> pd.Series:
        """Creates $1M cash only portfolios.
        A series of all 0s except at the USDOLLAR entry in the final position.
        """
        portfolio = pd.Series([0 for _ in self.data.tickers] + [1_000_000])
        portfolio.index = np.append(self.data.tickers, "USDOLLAR")
        return portfolio

    def h_return(self, h: pd.Series) -> float:
        """Calculates expected return for a portfolio produced by self.execute().

        :param h: Portfolio to calculate return for
                  Cash position ignored

        :return: Expected return at current time
        """
        w = h / np.sum(np.abs(h))  # Portfolio by asset weight
        exp_returns = self._default_r_forecast().estimate(self.data, self.t)
        # Include risk-free rate for cash position
        # exp_returns["USDOLLAR"] = self.risk_free_rate
        return 1 + sum(map(lambda a, b: a * b, exp_returns, w))

    def h_risk(self, h: pd.Series) -> float:
        """Calculates expected risk for a portfolio produced by self.execute().

        :param h: Portfolio to calculate risk for
                  Must not contain the cash position

        :return: Expected risk at current time
        """
        w = h / np.sum(np.abs(h))  # Portfolio by asset weight
        risk_mat = self._default_risk_metric().estimate(self.data, self.t)
        return w.T @ risk_mat @ w

    def execute(self, h: pd.Series, t: pd.Timestamp = None, *args,
                r_target: None | float = None,
                sig_thresh: None | float = None) -> list[TradeResult]:
        """Executes all trading policies at current or user specified time.

        :param h: Holdings vector, in dollars, including the cash account (the last element).
        :param t: Time of execution
                  Defaults to now

        :param r_target: Returns target value
        :param sig_thresh: Risk threshold value
                           SHOULD NOT BE USED! RISK THRESHOLD DOESN'T WORK

        :return: List of trade weights, trade timestamps, and shares traded
        """
        addtl_constraints = []

        if t is None:
            t = self.t
        if r_target is not None:
            rhat = self._default_r_forecast().estimate(self.data, self.t)
            addtl_constraints.append(ReturnsTarget(rhat, r_target))
        if sig_thresh is not None:
            # TODO WHY IS THE VARCOVAR MATRIX NOT POSITIVE DEFINITE
            sigma = self._default_risk_metric().estimate(self.data, self.t)
            addtl_constraints.append(RiskThreshold(sigma, sig_thresh))

        policies = [
            self._make_policy(gr, gt, addtl_constraints)
            for gr in [5, 10, 20, 50, 100, 200, 500]
            for gt in [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5]
        ]

        return list(map(lambda p: p.execute(h, self.data, t), policies))
