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

Additonally, to install this as a python package, run
```bash
# Install this as a editable dependency for the python REPL
# The python package itself is called starline_optimizer.
pip install -e .
```

The flagship product of this repo is `starline_optimizer.OptimizationEngine`,  
built around [pypfopt.efficient_frontier.EfficientFrontier](https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html) of PyPortfolioOpt.  
```python
# Sample code
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
