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
    t: pd.Timestamp  # Trade execution time (usually today)
    risk_free_rate: float

    def __init__(self, tickers: list[str], id: str, returns: pd.DataFrame | None,
                 varcovar: pd.DataFrame | None):
        """
        :param tickers: Tickers to optimize a portfolio over
        :param id: Job uuid
        :param returns: Forward looking returns
                        returns.index[0] (the first DataFrame row timestamp) should be today
        :param varcovar: Ticker variance-covariance matrix
        """
        self.tickers = tickers
        self.returns = returns
        self.varcovar = varcovar
        self.__id = id

        self.data = DataProvider(tickers, id, returns)
        self.t = self.data.trading_calendar()[0]  # Represents when we execute trades, ie. t = today
        self.risk_free_rate = 1.04 ** (1/12)  # TODO temp non-annualized risk free rate

        init_log = f"{self.__id} Successfully initalized OptimizationEngine with {self.tickers}"
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

    def _default_r_forecast(self):
        """Produces a new returns forecaster instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.ReturnsForecast(r_hat=self.returns, decay=0.5)

    def _default_risk_metric(self):
        """Produces a new risk metric instance.
        The same forecast instance can't be used multiple times in convex equation solvers.
        """
        return cvx.FullCovariance(Sigma=self.varcovar)

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
            self._default_r_forecast()
            - gamma_risk * self._default_risk_metric()
            - gamma_trade * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1), *constraints],
            planning_horizon=1,
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
        exp_returns = self.returns.iloc[0].to_numpy()
        # Append risk-free rate for cash position
        np.append(exp_returns, self.risk_free_rate)
        return 1 + sum(map(lambda a, b: a * b, exp_returns, w))

    def h_risk(self, h: pd.Series) -> float:
        """Calculates expected risk for a portfolio produced by self.execute().

        :param h: Portfolio to calculate risk for
                  Must not contain the cash position

        :return: Expected risk at current time
        """
        w = h / np.sum(np.abs(h))  # Portfolio by asset weight
        risk_mat = self.varcovar.to_numpy()
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
        policies = [
            self._make_policy(gr, gt, addtl_constraints)
            for gr in risk_gammas for gt in trade_gammas
        ]
        self._log(logger.info, f"{self.__id} Executing {len(policies)} MPO policies")

        res = list(map(lambda p: p.execute(h, self.data, t), policies))
        self._log(logger.success, f"{self.__id} MPO execution succeeded")
        return res
