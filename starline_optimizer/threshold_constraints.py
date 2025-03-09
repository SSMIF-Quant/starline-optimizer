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

    def compile_to_cvxpy(self, w_plus: cp.Variable, z: cp.Variable,
                         *args, **kwargs) -> cp.Constraint:
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :param z: Trade weights.
        """
        lim_param = cp.Parameter()
        lim_param.value = (self.lim ** (1/252)) - 1  # De-annualize returns threshold

        exp_rhat = cp.Parameter(len(self.rhat))
        exp_rhat.value = np.array(self.rhat)
        return exp_rhat.T @ w_plus[:-1] >= lim_param
