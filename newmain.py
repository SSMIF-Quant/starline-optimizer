import cvxportfolio as cvx

# Initialize Policy
current_period_returns_forecast = # historical mean return forecast
next_period_returns_forecast = # historical mean return forecast for the next period

gamma_risk = cvx.Gamma(initial_value = 0.5)
gamma_hold = cvx.Gamma(initial_value = 1.0)
gamma_trade = cvx.Gamma(initial_value = 1.0)

objective_1 = cvx.ReturnsForecast(r_hat = current_period_returns_forecast) \
    - gamma_risk * cvx.FullCovariance() \
    - gamma_hold * cvx.HoldingCost(short_fees = 1.) \
    - gamma_trade * cvx.TransactionCost(a = 2E-4)

objective_2 = cvx.ReturnsForecast(r_hat = next_period_returns_forecast) \
    - gamma_risk * cvx.FullCovariance() \
    - gamma_hold * cvx.HoldingCost(short_fees = 1.) \
    - gamma_trade * cvx.TransactionCost(a = 2E-4)

constraints_1 = [cvx.LongOnly(applies_to_cash = True)]  #Example
constraints_2 = [cvx.LeverageLimit(1)] #Example

policy = cvx.MultiPeriodOptimization(
    objective = [objective_1, objective_2],
    constraints = [constraints_1, constraints_2]
)

# 