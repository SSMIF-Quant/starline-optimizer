import cvxportfolio as cvx
import pandas as pd
from datetime import timedelta

from starline_optimizer.data_provider import DataProvider

def main():
    # Initialize tickers we want to optimize for
    tickers = ["XLB", "XLE", "XLC", "XLF",  "XLK", "XLRE", "XLU", "XLV", "XLI", "XLP", "XLY"]  # Add your desired tickers

    # Create DataProvider instance which will automatically:
    # 1. Update timeseries data for each ticker from Clickhouse
    # 2. Get historical prices and volumes
    # 3. Handle missing data through forward-filling
    # 4. Calculate returns
    data_provider = DataProvider(tickers)

    print(data_provider.tickers)
    # Get the trading calendar (all available trading dates)
    trading_calendar = data_provider.trading_calendar()
    
    # Make sure we have at least 365 days of data before starting
    lookback_days = 365
    test_days = 252 * 2  # 2 years of trading days (252 trading days per year)
    
    if len(trading_calendar) <= lookback_days + test_days:
        raise ValueError(f"Not enough trading days available. Need at least {lookback_days + test_days} days.")
    
    first_trading_date = trading_calendar[lookback_days]
    end_trading_date = trading_calendar[lookback_days + test_days]
    
    print(f"Starting backtest at {first_trading_date}")
    print(f"Ending backtest at {end_trading_date}")
    print(f"Total trading days being tested: {test_days}")
    
    # Calculate historical mean returns for our forecasts
    past_returns, current_returns, past_volumes, current_volumes, current_prices = data_provider.serve(first_trading_date)
    
    # Check that we have enough historical data
    if len(past_returns) < lookback_days - 10:  # Allow for a few missing days
        print(f"Warning: Only have {len(past_returns)} days of history, which is less than expected {lookback_days}")
    
    # Use past returns to create our simple forecast
    current_period_returns_forecast = past_returns.mean()
    next_period_returns_forecast = current_period_returns_forecast  # Using same forecast for next period

    # Create symbolic gamma parameters for optimization
    gamma_risk = cvx.Gamma(initial_value=0.5)
    gamma_hold = cvx.Gamma(initial_value=1.0)
    gamma_trade = cvx.Gamma(initial_value=1.0)

    # We will have to review this later to figure out if it is the objective we actually want to
    # use, to incorporate our factor matrix. The covariance matrix is already implemented within
    # the optimizer.
    objective_1 = cvx.ReturnsForecast(r_hat=current_period_returns_forecast) \
        - gamma_risk * cvx.FullCovariance() \
        - gamma_hold * cvx.HoldingCost(short_fees=1.) \
        - gamma_trade * cvx.TransactionCost(a=2E-4)

    # placeholder objective 2
    objective_2 = cvx.ReturnsForecast(r_hat=next_period_returns_forecast) \
        - gamma_risk * cvx.FullCovariance() \
        - gamma_hold * cvx.HoldingCost(short_fees=1.) \
        - gamma_trade * cvx.TransactionCost(a=2E-4)

    # Constraints (so far that I've thought of)
    # long only
    # limited rebalances
    # leverage limit
    # not a constraint, but our current cash position would impact our integer-rounding for each equity trade.

    # placeholderexample constraints
    constraints_1 = [cvx.LongOnly(applies_to_cash=True)]  #Example
    constraints_2 = [cvx.LeverageLimit(1)] #Example

    # the MultiPeriodOptimization policy combines the objectives and constraints into an MPO. Quite simple.
    policy = cvx.MultiPeriodOptimization(
        objective=[objective_1, objective_2],
        constraints=[constraints_1, constraints_2]
    )

    # Create market simulator
    simulator = cvx.StockMarketSimulator(
        market_data=data_provider,
        costs=[
            cvx.StocksTransactionCost(),
            cvx.StocksHoldingCost()
        ],
        round_trades=True,
        max_fraction_liquidity=0.05
    )

    # Optimize hyperparameters
    print("\nOptimizing hyperparameters...")
    optimized_policy = simulator.optimize_hyperparameters(
        policy=policy,
        start_time=first_trading_date,
        end_time=end_trading_date,  # Use shorter period
        initial_value=1_000_000,
        objective='sharpe_ratio'
    )

    # Print the optimized gamma values
    print("\nOptimized hyperparameters:")
    print(f"gamma_risk: {gamma_risk._index:.3f}")
    print(f"gamma_hold: {gamma_hold._index:.3f}")
    print(f"gamma_trade: {gamma_trade._index:.3f}")

    # Run backtest with optimized policy
    print("\nRunning backtest with optimized parameters...")
    result = simulator.backtest(
        policy=optimized_policy,
        start_time=first_trading_date,
        end_time=end_trading_date,  # Use same shorter period
        initial_value=1_000_000
    )

    # Print backtest results
    print("\nBacktest Results:")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Average Return: {result.average_return:.2%}")
    print(f"Annualized Average Return: {result.annualized_average_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Annualized Volatility: {result.annualized_volatility:.2%}")
    print(f"Maximum Drawdown: {max(result.drawdown):.2%}")
    print(f"Total Profit: ${result.profit:,.2f}")
    print(f"Average Turnover: {result.turnover.mean():.2%}")
    
    # Print portfolio weights for each period
    print("\nPortfolio Weights Over Time:")
    weights_df = result.w  # Get the weights DataFrame
    print("\nFirst 5 periods:")
    print(weights_df.head())
    print("\nLast 5 periods:")
    print(weights_df.tail())
    
    # Print average weights for each asset
    print("\nAverage Portfolio Weights:")
    print(weights_df.mean())
    
    # Save weights to CSV for further analysis
    weights_df.to_csv('portfolio_weights.csv')
    print("\nFull weights history saved to 'portfolio_weights.csv'")
    
    # Plot results
    result.plot()
    
    # Optionally, plot execution times
    result.times_plot()

if __name__ == "__main__":
    main()

    # 