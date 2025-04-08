import time
import json
import numpy as np
import pandas as pd
import cvxportfolio as cvx
from typing import Callable

from .env import APP_ENV
from .logger import logger
from .data_provider import DataProvider
from .threshold_constraints import ReturnsTarget, RiskThreshold

# u, t, shares_traded
type TradeResult = tuple[pd.Series, pd.Timestamp, pd.Series]

# expected return, expected risk
type PortfolioPerformance = tuple[float, float]


class OptimizationEngine:
    __id: str
    data: DataProvider
    tickers: list[str]
    t: pd.Timestamp
    risk_free_rate: float

    def __init__(self, tickers: list[str]):
        # TODO get return forecasts from clickhouse and pass into r_forecast
        self.tickers = tickers
        self.data = DataProvider(tickers)
        self.t = self.data.trading_calendar()[-1]  # Current trading time
        self.risk_free_rate = 1.04 ** (1/252)  # TODO temp non-annualized risk free rate
        self._genid()
        self._log(logger.info, "Successfully initalized {self.__id} with tickers {self.tickers}")

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
                "class_instance": self.__id,
                "data_instance": self.data.__id,
                "tickers": self.tickers,
                "message": message,
                **addtl_fields
                }))
        else:
            if addtl_fields == {}:
                severity(message)
            else:
                severity(f"{message}\n{json.dumps(addtl_fields, indent=4)}")

    def _genid(self):
        """Generates an 8-digit hash for the __id field of this OptimizationEngine. """
        hashstr = str(self.tickers) + str(time.time())
        self.__id = "OptimizationEngine" + str(abs(hash(hashstr)) % (10 ** 8))

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
        self._log(logger.trace, f"{self.__id} Created new MPO policy", {
            "gamma_risk": gamma_risk,
            "gamma_trade": gamma_trade,
            "constraints": str(constraints),
            "r_forecaster": str(self._default_r_forecast())
            })
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
        if r_target is not None:  # Append returns target constraint if r_target exists
            rhat = self._default_r_forecast().estimate(self.data, self.t)
            addtl_constraints.append(ReturnsTarget(rhat, r_target))
        if sig_thresh is not None:  # Append risk threshold constraint if sig_thresh exists
            sigma = self._default_risk_metric().estimate(self.data, self.t)
            addtl_constraints.append(RiskThreshold(sigma, sig_thresh))

        risk_gammas = [5, 10, 20, 50, 100, 200, 500]
        trade_gammas = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5]
        self._log(logger.info, f"{self.__id} Executing MPO policies", {
            "risk_gammas": risk_gammas,
            "trade_gammas": trade_gammas
            })
        policies = [
            self._make_policy(gr, gt, addtl_constraints)
            for gr in risk_gammas for gt in trade_gammas
        ]

        res = list(map(lambda p: p.execute(h, self.data, t), policies))
        self._log(logger.success, f"{self.__id} MPO execution succeeded")
        return res
