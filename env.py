from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np

from model import nl_dynamics_local


class LtiNetwork(gym.Env[np.ndarray, np.ndarray]):
    """A discrete time network of lti systems."""

    x_bnd = np.array([[0, -1], [1, 1]])
    u_bnd = np.array([[-1], [1]])
    e_bnd = np.array([[-0.1], [0]])

    w = np.array([[5e2, 5e2]])

    gamma = 0.9

    def __init__(self, n: int, Adj: np.ndarray):
        super().__init__()
        self.n = n
        self.nx = 2 * n
        self.x_bnd = np.tile(self.x_bnd, n)
        self.u_bnd = np.tile(self.u_bnd, n)
        self.w = np.tile(self.w, (1, n))
        self.Adj = Adj

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the state of the network. An x0 can be passed in the options dict."""
        super().reset(seed=seed, options=options)
        if options is not None and "x0" in options:
            self.x = options["x0"]
        else:
            self.x = np.tile([0, 0.15], self.n).reshape(self.nx, 1)
        return self.x, {}

    def get_stage_cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """Computes the stage cost `L(s,a)`."""
        lb, ub = self.x_bnd
        return 0.5 * float(
            np.square(state).sum()
            + 0.5 * np.square(action).sum()
            + self.w @ np.maximum(0, lb[:, np.newaxis] - state)
            + self.w @ np.maximum(0, state - ub[:, np.newaxis])
        )

    def get_dist_stage_cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """Computes the distributed stage cost `L(s,a)`."""
        lb, ub = self.x_bnd
        x = np.split(state, self.n)
        u = np.split(action, self.n)
        lb = np.split(lb, self.n)
        ub = np.split(ub, self.n)
        w = np.split(self.w, self.n, axis=1)
        return [
            0.5
            * float(
                np.square(x[i]).sum()
                + 0.5 * np.square(u[i]).sum()
                + w[i] @ np.maximum(0, lb[i][:, np.newaxis] - x[i])
                + w[i] @ np.maximum(0, x[i] - ub[i][:, np.newaxis])
            )
            for i in range(self.n)
        ]

    def step(
        self, action: cs.DM
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Steps the system."""
        if isinstance(action, cs.SX):
            action = cs.DM(action)
        r = self.get_stage_cost(self.x, action)
        r_dist = self.get_dist_stage_cost(self.x, action)
        u = np.split(action, self.n)
        x = np.split(self.x, self.n)
        e = self.np_random.uniform(*self.e_bnd, size=(self.n, 1))
        x_new = [
            nl_dynamics_local(
                x[i], u[i], [x[j] for j in range(self.n) if self.Adj[i, j]], e[i], {}
            )
            for i in range(self.n)
        ]
        x_new = np.concatenate(x_new, axis=0).reshape(self.nx, 1)
        self.x = x_new
        return x_new, r, False, False, {"r_dist": r_dist}
