from typing import Iterable
import cvxportfolio as cvx
import cvxpy as cp
import numpy as np


class ReturnsTarget(cvx.constraints.Constraint):
    # TODO dataframe of returns
    # TODO include_cash option
    def __init__(self, rhat: Iterable[float], lim: float):
        """Returns target constraint for the convex optimizer.
        Ensures all portfolios have returns greater than or equal to lim.

        :param rhat: Expected returns for next trading period
        :param lim: Annualized portfolio returns target value
        """
        self.rhat = rhat
        self.lim = lim
        return

    def __str__(self):
        return repr(self)

    def __repr__(self):
        # TODO rhat shouldn't be omitted from repr
        return f"ReturnsTarget({self.lim})"

    def compile_to_cvxpy(
        self, w_plus: cp.Variable, z: cp.Variable, *args, **kwargs
    ) -> cp.Constraint:
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :param z: Trade weights.
        """
        lim_param = cp.Parameter()
        lim_param.value = (self.lim ** (1 / 252)) - 1  # De-annualize returns target

        exp_rhat = cp.Parameter(len(self.rhat))
        exp_rhat.value = np.array(self.rhat)
        return exp_rhat.T @ w_plus[:-1] >= lim_param


# TODO completely broken due to cxvpy and how it handles the quadratic form
class RiskThreshold(cvx.constraints.Constraint):
    def __init__(self, sigma: Iterable[Iterable[[float]]], lim: float):
        """Risk threshold constraint for the convex optimizer.
        Ensures all portfolios have risk less than or equal to lim.

        :param sigma: (Square) risk matrix
        :param lim: Annualized portfolio risk threshold value
        """
        print("[WARN] RiskThreshold is completely unusable.")
        self.sigma = sigma
        self.lim = lim
        return

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"RiskThreshold({self.lim})"

    def compile_to_cvxpy(
        self, w_plus: cp.Variable, z: cp.Variable, *args, **kwargs
    ) -> cp.Constraint:
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :param z: Trade weights.
        """
        active_assets = w_plus[:-1]  # Ignore cash position when calculating risk
        lim_param = cp.Parameter()
        lim_param.value = (self.lim / 252) - 1  # De-annualize risk threshold

        exp_risk = cp.Parameter(self.sigma.shape)
        exp_risk.value = np.array(self.sigma)
        return active_assets.T @ exp_risk @ active_assets <= lim_param
