from typing import Callable
import pandas as pd
import yfinance as yf
import pypfopt

# TODO Recover the Transaction costs and Holding costs from cvxportfolio
# TODO Multi-period optimization (reconstruct returns, risk model, etc.)
# TODO Risk metric integration; we can't just supply a DataFrame of risk calculations


class OptimizationEngine:
    data: pd.DataFrame
    returns: pd.Series
    risk_model: pd.DataFrame
    optimizer: Callable[[], pypfopt.efficient_frontier.EfficientFrontier]

    def __init__(self, assets: list[str]):
        # TODO get returns/risk metrics from clickhouse
        self.data = yf.Tickers(assets).download()["Close"]
        self.returns = pypfopt.expected_returns.mean_historical_return(self.data)
        self.risk_model = pypfopt.risk_models.risk_matrix(self.data)
        self.optimizer = pypfopt.efficient_frontier.EfficientFrontier

    def efficient_risk(self, risk: float):
        """Creates an optimized portfolio for the risk threshold.

        :param risk: Max risk metric to avoid, must be positive.
        """
        return self.optimizer(self.returns, self.risk_model).efficient_risk(risk)

    def efficient_return(self, returns: float):
        """Creates an optimized portfolio for some target return value.

        :param returns: Required return of the resulting portfolio.
        """
        return self.optimizer(self.returns, self.risk_model).efficient_return(returns)
