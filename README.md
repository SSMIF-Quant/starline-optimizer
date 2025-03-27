# starline-optimizer

### SSMIF convex portfolio optimization program built using cvxportfolio.
---
</br>

## Quickstart
To get started, clone the repository and do
```bash
cd starline-optimizer/  # cd into this repo
python -m venv .venv  # Create a python virtualenv to download starline-optimizer and its dependencies
source .venv/bin/activate  # source into the virtualenv
pip install -r requirements.txt  # Install required dependencies for starline-optimizer
```

In the event of ERROR: Can not perform a '--user' install. User site-packages are not visible in this virtualenv.

Set BELOW in pyvenv.cfg
include-system-site-packages = false --> include-system-site-packages = true

Additonally, to install this as a python package, run
```bash
# Install this as a editable dependency for the python REPL
# The python package itself is called starline_optimizer.
pip install -e .
```

The flagship product of this repo is `starline_optimizer.OptimizationEngine`,  
built around [cvxportfolio.MultiPeriodOptimization](https://www.cvxportfolio.com/en/stable/optimization_policies.html#cvxportfolio.MultiPeriodOptimization).  
As of now, the optimizer takes a list of Yahoo finance tickers, and retrieves data about those tickers from yfinance.  
```python
# Example code
from starline_optimizer import OptimizationEngine
op = OptimizationEngine(["AAPL", "IBM", "MSFT"])
u, t, shares_traded = op.execute(op._cash_only())[0]
print("\nPortfolio trade weights:\n", u)
print("\nTrade execution time:\n", t)
print("\nShares traded:\n", shares_traded)
```

---

## Repo Structure
Root level includes most of the python build tools.  
Source code is found in `./starline_optimizer`.  
`starline_optimizer.OptimizationEngine` can be found in `./starline_optimizer/engine.py`.  
Data is fed to the optimizer through `DataProvider` in `./starline_optimizer/data_provider.py`  

---
