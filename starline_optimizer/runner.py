from typing import ReadOnly
import cvxportfolio as cvxp
import pandas as pd
from cvxpy import SCS


class OptimizationEngine:
    assets: ReadOnly[list[str]]
    backtest_sim: ReadOnly[cvxp.StockMarketSimulator]
    policies: ReadOnly[None | list[cvxp.policies.Policy]]

    def __init__(self, assets: list[str]):
        self.assets = assets
        self.backtest_sim = cvxp.StockMarketSimulator(
            assets, trading_frequency="weekly"
        )
        self.policies = None
        self.results = None

    def _make_policy(self, gr: float, gt: float) -> cvxp.policies.Policy:
        """Creates an optimization policy with the provided hyperparameters.

        :param gr: gamma-risk; risk aversion hyperparameter.
        :param gt: gamma-trade; trade aversion hyperparameter.
        """
        return cvxp.MultiPeriodOptimization(
            cvxp.ReturnsForecast()  # TODO get these from clickhouse
            - gr * cvxp.FactorModelCovariance(num_factors=10)
            - gt * cvxp.StocksTransactionCost(),
            [cvxp.LongOnly(), cvxp.LeverageLimit(1)],
            planning_horizon=6,
            solver=SCS,
        )

    def make_policies(self) -> list[cvxp.policies.Policy]:
        """Creates many optimization policies with different hyperparameters.
        The policies class variable is populated with the output of this function.
        """
        policies = [
            self._make_policy(gr, gt)
            for gr in [5, 10, 50, 100, 500, 1000]
            for gt in [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        ]
        self.policies = policies
        return policies

    def _cash_portfolio(self) -> pd.Series:
        """Generates a cash portfolio over the provided assets """
        cash_h = pd.Series([0 for _ in self.assets[0:-1]] + [1000000])
        cash_h.index = self.assets
        return cash_h

    def _calculate_trade(self, policy: cvxp.policies.Policy) -> pd.Series:
        """Creates a portfolio based on a trading policy. """
        # Cash return is unknown at t = -1 (today)
        # Which means these executions are using last week's data
        data = self.backtest_sim.market_data
        trade_t = data.trading_calendar()[-1]
        assets = data[trade_t]
        print(assets)
        return

        # TODO variable initial portfolio
        # initial trading portfolio is 1mil of all cash
        cash_h = self._cash_portfolio()
        return policy.execute(cash_h, self.backtest_sim.market_data, trade_t)

    def calculate_trades(self) -> list[pd.Series]:
        """Creates portfolios based on all the policies in self.policies.

        :param assets: The assets to create a portfolios over
        """
        return list(map(lambda p: self._calculate_trade(p), self.policies))
