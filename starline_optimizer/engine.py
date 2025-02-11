from typing import ReadOnly
import pandas as pd
import cvxportfolio as cvx
from .data_provider import DataProvider


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
