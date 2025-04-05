import cvxportfolio as cvx
import pandas as pd
from datetime import timedelta

from starline_optimizer.data_provider import DataProvider

def main():
    # Initialize tickers we want to optimize for
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Add your desired tickers
        # Create DataProvider instance which will automatically:
    # 1. Update timeseries data for each ticker from Clickhouse
    # 2. Get historical prices and volumes
    # 3. Handle missing data through forward-filling
    # 4. Calculate returns
    # Create DataProvider instance
    data_provider = DataProvider(tickers)

    print(data_provider.tickers)
    # Get the trading calendar (all available trading dates)
    trading_calendar = data_provider.trading_calendar()
    
    # Make sure we have at least 365 days of data before starting
    # This ensures we have enough historical data to calculate means and covariances
    lookback_days = 365
    
    # Find a suitable starting point with enough history
    if len(trading_calendar) <= lookback_days:
        raise ValueError(f"Not enough trading days available. Need at least {lookback_days} days.")
    
    # Set the first trading date to be lookback_days into the calendar
    first_trading_date = trading_calendar[lookback_days]
    print(f"Starting backtest at {first_trading_date} with {lookback_days} days of history")
    
    # Calculate historical mean returns for our forecasts
    past_returns, current_returns, past_volumes, current_volumes, current_prices = data_provider.serve(first_trading_date)
    
    # Check that we have enough historical data
    if len(past_returns) < lookback_days - 10:  # Allow for a few missing days
        print(f"Warning: Only have {len(past_returns)} days of history, which is less than expected {lookback_days}")
    
    # Use past returns to create our simple forecast
    current_period_returns_forecast = past_returns.mean()
    next_period_returns_forecast = current_period_returns_forecast  # Using same forecast for next period

    # Use fixed gamma values 
    gamma_risk = 0.5
    gamma_hold = 1.0
    gamma_trade = 1.0

    # Define objectives with fixed gamma values
    objective_1 = cvx.ReturnsForecast(r_hat=current_period_returns_forecast) \
        - gamma_risk * cvx.FullCovariance() \
        - gamma_hold * cvx.HoldingCost(short_fees=1.) \
        - gamma_trade * cvx.TransactionCost(a=2E-4)

    objective_2 = cvx.ReturnsForecast(r_hat=next_period_returns_forecast) \
        - gamma_risk * cvx.FullCovariance() \
        - gamma_hold * cvx.HoldingCost(short_fees=1.) \
        - gamma_trade * cvx.TransactionCost(a=2E-4)

    constraints_1 = [cvx.LongOnly(applies_to_cash=True)]
    constraints_2 = [cvx.LeverageLimit(1)]

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

    # Run backtest starting after we have enough history
    print("\nRunning backtest...")
    result = simulator.backtest(
        policy=policy,
        start_time=first_trading_date,
        end_time=trading_calendar[-1],
        initial_value=1_000_000
    )

    # Print backtest results
    print("\nBacktest Results:")
    try:
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Average Return: {result.average_return:.2%}")
        print(f"Annualized Average Return: {result.annualized_average_return:.2%}")
        print(f"Volatility: {result.volatility:.2%}")
        print(f"Annualized Volatility: {result.annualized_volatility:.2%}")
        print(f"Maximum Drawdown: {max(result.drawdown):.2%}")
        print(f"Total Profit: ${result.profit:,.2f}")
        print(f"Average Turnover: {result.turnover.mean():.2%}")
    except (TypeError, AttributeError) as e:
        print(f"Error calculating metrics: {e}")
    
    # Plot results (this will show portfolio value, weights, and other metrics)
    result.plot()
    
    # Optionally, plot execution times
    result.times_plot()

if __name__ == "__main__":
    main()

    # 