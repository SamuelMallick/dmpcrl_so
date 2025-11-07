from typing import Optional

import numpy as np
from mpcrl.optim import NewtonMethod


class CentralizedSecondOrderOptimizer(NewtonMethod):
    # centralized second-order update regularized the same as the distributed one
    # for comparison

    def _second_order_update(self, gradient, hessian) -> tuple:
        theta = self.learnable_parameters.value
        lr = self.lr_scheduler.value
        solver = self._update_solver
        if solver is None:
            # regularization 1e-3 applied for non-singularity
            # a error will be raised here, from numpy.linalg.solve, if the matrix is singular
            dtheta = -lr * np.linalg.solve(
                hessian + 1e-3 * np.eye(hessian.shape[0]), gradient
            )
            return theta + dtheta, None
        raise NotImplementedError("Not implemented bro!")


class DistributedSecondOrderOptimizer(NewtonMethod):
    C = None  # matrix that is agreed upon with consensus
    sigma = None  # regularization parameter

    def update(
        self,
        gradient,
        hessian,
        info: Optional[dict] = None,
    ) -> Optional[str]:
        if self._order == 1:
            theta_new, status = self._first_order_update(gradient)
        else:
            theta_new, status = self._second_order_update(gradient, hessian, info)
        if self.bound_consistency:
            theta_new = np.clip(
                theta_new, self.learnable_parameters.lb, self.learnable_parameters.ub
            )
        self.learnable_parameters.update_values(theta_new)
        return status

    def _second_order_update(self, gradient, hessian, info) -> tuple:
        solver = self._update_solver
        if solver is None:
            theta = self.learnable_parameters.value
            lr = self.lr_scheduler.value
            sigma_i = self.sigma
            C = self.C
            if C is None or sigma_i is None:
                raise ValueError(
                    f"`{self.__class__.__name__}` optimizer requires consensus matrix C and regularization sigma (computed by coordinator) to perform second order updates."
                )

            gs = info["dQs"]
            td_errors = info["td_errors"]
            hs = info["hs"]
            K_i = np.mean(hs, 0)

            T = len(gs)
            K_tilde_i = np.linalg.inv(T * sigma_i * np.eye(gs[0].shape[0]) - K_i)
            G_i = np.array(gs).T
            delta = np.array(td_errors).reshape(-1, 1)
            M = np.linalg.inv(np.eye(T) + C)
            dtheta = lr * (K_tilde_i @ G_i @ (delta - M @ C @ delta))

            self.C, self.sigma = None, None  # reset after use
            return (
                theta
                + dtheta.reshape(
                    -1,
                ),
                None,
            )
        # second order update not implemented for constrained case
        raise NotImplementedError(
            f"`{self.__class__.__name__}` optimizer does not implement `_second_order_update` with constraints."
        )
