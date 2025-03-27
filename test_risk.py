from starline_optimizer import OptimizationEngine

def test_optimization():
    # Initialize optimizer with our stocks
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    op = OptimizationEngine(stocks)
    
    # Get initial cash-only portfolio
    h = op._cash_only()
    
    # Run optimization with a 6% annual return target
    portfolios = op.execute(h, r_target=1.06)
    
    # Calculate annualized returns for each portfolio
    returns = list(map(lambda p: op.h_return(p[0] + h), portfolios))
    annualized_returns = list(map(lambda r: r ** 252, returns))
    
    # Calculate annualized risks
    risks = list(map(lambda p: op.h_risk(p[0][:-1]) * 252, portfolios))
    
    # Calculate Sharpe ratios (using 4% risk-free rate)
    risk_free_rate = 1.04
    sharpes = list(map(lambda r, sig: (r - risk_free_rate) / sig, annualized_returns, risks))
    
    # Print results
    print("\nOptimization Results:")
    print("\n--- Portfolio Allocations ---")
    for i, (u, t, shares) in enumerate(portfolios):
        print(f"\nPortfolio {i+1}:")
        for asset, value in zip(stocks + ['USDOLLAR'], u):
            print(f"{asset}: ${value:,.2f}")
        
    print("\n--- Expected Annualized Returns ---")
    for i, r in enumerate(annualized_returns):
        print(f"Portfolio {i+1}: {r:.2%}")
    
    print("\n--- Annualized Risk ---")
    for i, risk in enumerate(risks):
        print(f"Portfolio {i+1}: {risk:.2%}")
        
    print("\n--- Sharpe Ratios ---")
    for i, sharpe in enumerate(sharpes):
        print(f"Portfolio {i+1}: {sharpe:.2f}")

if __name__ == "__main__":
    test_optimization()