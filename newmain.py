import cvxportfolio as cvx

from starline_optimizer.data_provider import DataProvider

def main():
    # Initialize tickers we want to optimize for
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Add your desired tickers

    # Create DataProvider instance which will automatically:
    # 1. Update timeseries data for each ticker from Clickhouse
    # 2. Get historical prices and volumes
    # 3. Handle missing data through forward-filling
    # 4. Calculate returns
    data_provider = DataProvider(tickers)

    print(data_provider.tickers)
    # Get the trading calendar (all available trading dates)
    trading_calendar = data_provider.trading_calendar()

    # Calculate historical mean returns for our forecasts
    # We'll use the first trading date to get all historical data
    first_trading_date = trading_calendar[0]
    past_returns, current_returns, past_volumes, current_volumes, current_prices = data_provider.serve(first_trading_date)

    # Use past returns to create our simple forecast
    current_period_returns_forecast = past_returns.mean()
    next_period_returns_forecast = current_period_returns_forecast  # Using same forecast for next period

    # Instead of fixed gamma values, let's use symbolic ones for optimization
    gamma_risk = cvx.Gamma(initial_value=0.5)  # initial value is just a starting point
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

    # Add backtesting functionality here
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

    # Add hyperparameter optimization
    print("Optimizing gamma values...")
    optimized_policy = simulator.optimize_hyperparameters(
        policy=policy,
        start_time=first_trading_date,
        end_time=trading_calendar[-1],
        initial_value=1_000_000,
        objective='sharpe_ratio'  # optimize for maximum Sharpe ratio
    )

    # Print the optimized gamma values
    print("\nOptimized gamma values:")
    print(f"gamma_risk: {gamma_risk.value:.3f}")
    print(f"gamma_hold: {gamma_hold.value:.3f}")
    print(f"gamma_trade: {gamma_trade.value:.3f}")

    # Run backtest with optimized policy
    print("\nRunning backtest with optimized parameters...")
    result = simulator.backtest(
        policy=optimized_policy,
        start_time=first_trading_date,
        end_time=trading_calendar[-1],
        initial_value=1_000_000
    )

    # Print backtest results
    print("\nBacktest Results:")
    print(f"Sharpe Ratio: {result.sharpe_ratio():.2f}")
    print(f"Annual Return: {result.annual_return():.2%}")
    print(f"Annual Volatility: {result.annual_volatility():.2%}")
    print(f"Maximum Drawdown: {result.max_drawdown():.2%}")
    
    # Plot results
    result.plot()

if __name__ == "__main__":
    main()

    # 