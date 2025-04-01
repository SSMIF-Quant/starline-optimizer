import cvxportfolio as cvx

# Initialize Policy
current_period_returns_forecast = # insert historical mean return forecast here
next_period_returns_forecast = # insert historical mean return forecast for the next period here

# We will have to iterate through several randomly chosen constants for these gamma values.
# CVXPortfolio has an in-built backtester that can do this for us.
gamma_risk = cvx.Gamma(initial_value = 0.5)
gamma_hold = cvx.Gamma(initial_value = 1.0)
gamma_trade = cvx.Gamma(initial_value = 1.0)

# We will have to review this later to figure out if it is the objective we actually want to
# use, to incorporate our factor matrix. The covariance matrix is already implemented within
# the optimizer.
objective_1 = cvx.ReturnsForecast(r_hat = current_period_returns_forecast) \
    - gamma_risk * cvx.FullCovariance() \
    - gamma_hold * cvx.HoldingCost(short_fees = 1.) \
    - gamma_trade * cvx.TransactionCost(a = 2E-4)

# placeholder objective 2
objective_2 = cvx.ReturnsForecast(r_hat = next_period_returns_forecast) \
    - gamma_risk * cvx.FullCovariance() \
    - gamma_hold * cvx.HoldingCost(short_fees = 1.) \
    - gamma_trade * cvx.TransactionCost(a = 2E-4)

# Constraints (so far that I've thought of)
# long only
# limited rebalances
# leverage limit
# not a constraint, but our current cash position would impact our integer-rounding for each equity trade.

# placeholderexample constraints
constraints_1 = [cvx.LongOnly(applies_to_cash = True)]  #Example
constraints_2 = [cvx.LeverageLimit(1)] #Example

# the MultiPeriodOptimization policy combines the objectives and constraints into an MPO. Quite simple.
policy = cvx.MultiPeriodOptimization(
    objective = [objective_1, objective_2],
    constraints = [constraints_1, constraints_2]
)

# 