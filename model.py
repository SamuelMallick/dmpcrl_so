import casadi as cs
import numpy as np


def nl_dynamics_local(
    x: cs.SX, u: cs.SX, xc: list[cs.SX], e: float, parameters: dict
) -> cs.SX:
    if parameters:  # if to be used for learning, symbolic params are passed
        a11, a12, a21, a22 = (
            parameters["a11"],
            parameters["a12"],
            parameters["a21"],
            parameters["a22"],
        )
        b1, b2 = parameters["b1"], parameters["b2"]
        ac = parameters["ac"]
    else:  # if to be used as the true model, the true numbers are used
        a11, a12, a21, a22 = 0.9, 0.35, 0, 1.1
        b1, b2 = 0.0813, 0.2
        ac = -0.1
    x1_new = a11 * x[0] + a12 * x[1] + b1 * u + e
    x2_new = a21 * x[0] + a22 * x[1] + b2 * u + ac * sum(xc_[1] for xc_ in xc)
    return cs.vertcat(x1_new, x2_new)


def nl_dynamics_global(
    x: cs.SX, u: cs.SX, e: list[float], Adj: np.ndarray, parameters: list[dict] = []
) -> cs.SX:
    n = Adj.shape[0]
    if not parameters:
        parameters = [{} for _ in range(n)]
    x_l = [x[i * 2 : (i + 1) * 2] for i in range(n)]
    u_l = [u[i : i + 1] for i in range(n)]
    x_new = []
    for i in range(n):
        xc = [x_l[j] for j in range(n) if Adj[i, j] == 1]
        x_new.append(nl_dynamics_local(x_l[i], u_l[i], xc, e[i], parameters[i]))
    return cs.vertcat(*x_new)
