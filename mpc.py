import casadi as cs
import numpy as np
from csnlp import Nlp, Solution
from csnlp.wrappers import Mpc
from dmpcrl.mpc.mpc_admm import MpcAdmm

from model import nl_dynamics_global, nl_dynamics_local


class GlobalLearningMpc(Mpc):
    x_bnd = np.array([[0, -1], [1, 1]])
    u_bnd = np.array([[-1], [1]])

    w_init = np.array([[5e2, 5e2]])

    def __init__(self, prediction_horizon: int, n: int, Adj: np.ndarray, gamma: float):
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)
        self.n = n
        self.nx_l = 2
        self.nu_l = 1

        # parameters
        V0, f, x_lb, x_ub, b, Q, R, w, dynam_params = [], [], [], [], [], [], [], [], []
        for i in range(n):
            V0.append(self.parameter(f"V0_{i}", (1,)))
            f.append(self.parameter(f"f_{i}", (self.nx_l + self.nu_l, 1)))
            x_lb.append(self.parameter(f"x_lb_{i}", (self.nx_l,)))
            x_ub.append(self.parameter(f"x_ub_{i}", (self.nx_l,)))
            b.append(self.parameter(f"b_{i}", (self.nx_l, 1)))
            Q.append(self.parameter(f"Q_{i}", (self.nx_l, 1)))
            R.append(self.parameter(f"R_{i}", (self.nu_l, 1)))
            w.append(self.parameter(f"w_{i}", (1, self.nx_l)))
            dynam_params.append(
                {
                    "a11": self.parameter(f"a11_{i}", (1,)),
                    "a12": self.parameter(f"a12_{i}", (1,)),
                    "a21": self.parameter(f"a21_{i}", (1,)),
                    "a22": self.parameter(f"a22_{i}", (1,)),
                    "b1": self.parameter(f"b1_{i}", (1,)),
                    "b2": self.parameter(f"b2_{i}", (1,)),
                    "ac": self.parameter(f"ac_{i}", (1,)),
                }
            )
        V0 = sum(V0)
        x_lb = cs.vertcat(*x_lb)
        x_ub = cs.vertcat(*x_ub)
        b = cs.vertcat(*b)
        Q = cs.vertcat(*Q)
        R = cs.vertcat(*R)
        w = cs.horzcat(*w)

        self.learnable_pars_init = {}
        for i in range(n):
            self.learnable_pars_init[f"V0_{i}"] = np.zeros((1, 1))
            self.learnable_pars_init[f"f_{i}"] = np.zeros((self.nx_l + self.nu_l, 1))
            self.learnable_pars_init[f"x_lb_{i}"] = np.array([0, 0]).reshape(-1, 1)
            self.learnable_pars_init[f"x_ub_{i}"] = np.array([0, 0]).reshape(-1, 1)
            self.learnable_pars_init[f"b_{i}"] = np.zeros(2)
            self.learnable_pars_init[f"Q_{i}"] = np.ones((self.nx_l, 1))
            self.learnable_pars_init[f"R_{i}"] = np.ones((self.nu_l, 1))
            self.learnable_pars_init[f"w_{i}"] = self.w_init * np.ones((1, self.nx_l))
            self.learnable_pars_init[f"a11_{i}"] = np.array([1]).reshape(-1, 1)
            self.learnable_pars_init[f"a12_{i}"] = np.array([0.25]).reshape(-1, 1)
            self.learnable_pars_init[f"a21_{i}"] = np.array([0]).reshape(-1, 1)
            self.learnable_pars_init[f"a22_{i}"] = np.array([1]).reshape(-1, 1)
            self.learnable_pars_init[f"b1_{i}"] = np.array([0.0312]).reshape(-1, 1)
            self.learnable_pars_init[f"b2_{i}"] = np.array([0.25]).reshape(-1, 1)
            self.learnable_pars_init[f"ac_{i}"] = np.array([0]).reshape(-1, 1)

        x_bnd = np.tile(self.x_bnd, n)
        u_bnd = np.tile(self.u_bnd, n)
        x, _ = self.state("x", 2 * n)
        u, _ = self.action(
            "u", n, lb=u_bnd[0].reshape(-1, 1), ub=u_bnd[1].reshape(-1, 1)
        )
        s, _, _ = self.variable("s", (self.nx_l * n, prediction_horizon + 1), lb=0)

        # dynamics
        self.set_nonlinear_dynamics(
            lambda x, u: nl_dynamics_global(x, u, [0] * n, Adj, dynam_params) + b
        )

        # constraints
        self.constraint(f"x_lb", x_bnd[0] + x_lb - s, "<=", x)
        self.constraint(f"x_ub", x, "<=", x_bnd[1] + x_ub + s)

        x_l = cs.vertsplit_n(x, n)
        u_l = cs.vertsplit_n(u, n)
        gammapowers = cs.DM(gamma ** np.arange(prediction_horizon + 1)).T
        self.minimize(
            V0
            + sum(
                cs.sum2(f[i].T @ cs.vertcat(x_l[i][:, :-1], u_l[i])) for i in range(n)
            )
            + 0.5
            * (
                cs.sum2(gammapowers * (cs.sum1((Q * x) ** 2) + w @ s))
                + cs.sum2(gammapowers[:-1] * 0.5 * cs.sum1((R * u) ** 2))
            )
        )

        # use these options for ipopt solver
        # opts = {
        #     "expand": True,
        #     "print_time": False,
        #     "bound_consistency": True,
        #     "calc_lam_x": True,
        #     "calc_lam_p": False,
        #     "ipopt": {
        #         "max_iter": 1000,
        #         "sb": "yes",
        #         "print_level": 0,
        #     },
        # }
        opts = {
            "print_time": False,
            "record_time": True,
            "error_on_fail": False,
            "printLevel": "none",
            "jit": False,
        }
        self.init_solver(opts, solver="qpoases")


class LocalLearningMpc(MpcAdmm):

    x_bnd = np.array([[0, -1], [1, 1]])
    u_bnd = np.array([[-1], [1]])

    w_init = np.array([[5e2, 5e2]])

    learnable_pars_init = {
        "V0": np.zeros((1, 1)),
        "f": np.zeros(2 + 1),
        "x_lb": np.array([0, 0]).reshape(-1, 1),
        "x_ub": np.array([0, 0]).reshape(-1, 1),
        "b": np.zeros(2),
        "Q": np.ones((2, 1)),
        "R": np.ones((1, 1)),
        "w": w_init,
        "a11": np.array([1]).reshape(-1, 1),
        "a12": np.array([0.25]).reshape(-1, 1),
        "a21": np.array([0]).reshape(-1, 1),
        "a22": np.array([1]).reshape(-1, 1),
        "b1": np.array([0.0312]).reshape(-1, 1),
        "b2": np.array([0.25]).reshape(-1, 1),
        "ac": np.array([0]).reshape(-1, 1),
    }
    fixed_pars_init = {}

    def add_dynamics(
        self, x, x_c, u, b, prediction_horizon, num_neighbours, dynam_params
    ):
        x_c_list = cs.vertsplit_n(x_c, num_neighbours)
        for k in range(prediction_horizon):
            self.constraint(
                f"dyn_{k}",
                x[:, [k + 1]],
                "==",
                (
                    nl_dynamics_local(
                        x[:, k],
                        u[:, k],
                        [_x_c[:, k] for _x_c in x_c_list],
                        0,  # 0 for disturbance as unknown
                        dynam_params,
                    )
                    + b
                ),
            )

    def __init__(
        self,
        prediction_horizon: int,
        num_neighbours: int,
        my_index: int,
        gamma: float,
        rho: float = 0.5,
    ):
        """Instantiate local MPC.

        Parameters
        ----------
        prediction_horizon : int
            The prediction horizon for the MPC.
        num_neighbours : int
            The number of neighbours for this agent. Determines the dimension of the
            local copies of coupling states.
        my_index : int
            The index of this agent in its own augmented state vector.
        gamma : float
            The discount factor for the cost function.
        rho : float, optional
            The ADMM penalty parameter, by default 0.5
        """
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, prediction_horizon)
        self.nx_l = 2
        self.nu_l = 1
        self.num_neighbours = num_neighbours

        # parameters
        V0 = self.parameter("V0", (1,))
        f = self.parameter("f", (self.nx_l + self.nu_l, 1))
        x_lb = self.parameter("x_lb", (self.nx_l,))
        x_ub = self.parameter("x_ub", (self.nx_l,))
        b = self.parameter("b", (self.nx_l, 1))
        Q = self.parameter("Q", (self.nx_l, 1))
        R = self.parameter("R", (self.nu_l, 1))
        w = self.parameter("w", (1, self.nx_l))
        dynam_params = {
            "a11": self.parameter("a11", (1,)),
            "a12": self.parameter("a12", (1,)),
            "a21": self.parameter("a21", (1,)),
            "a22": self.parameter("a22", (1,)),
            "b1": self.parameter("b1", (1,)),
            "b2": self.parameter("b2", (1,)),
            "ac": self.parameter("ac", (1,)),
        }

        # opt vars
        x, x_c = self.augmented_state(num_neighbours, my_index, self.nx_l)
        u, _ = self.action("u", 1, lb=self.u_bnd[0], ub=self.u_bnd[1])
        s, _, _ = self.variable("s", (self.nx_l, prediction_horizon + 1), lb=0)

        # dynamics
        self.add_dynamics(
            x, x_c, u, b, prediction_horizon, num_neighbours, dynam_params
        )

        self.constraint(f"x_lb", self.x_bnd[0] + x_lb - s, "<=", x)
        self.constraint(f"x_ub", x, "<=", self.x_bnd[1] + x_ub + s)

        gammapowers = cs.DM(gamma ** np.arange(prediction_horizon + 1)).T
        self.set_local_cost(
            V0
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * (
                cs.sum2(gammapowers * (cs.sum1((Q * x) ** 2) + w @ s))
                + cs.sum2(gammapowers[:-1] * 0.5 * cs.sum1((R * u) ** 2))
            ),
            rho=rho,
        )

        # use these options for ipopt solver
        # opts = {
        #     "expand": True,
        #     "print_time": False,
        #     "bound_consistency": True,
        #     "calc_lam_x": True,
        #     "calc_lam_p": False,
        #     "ipopt": {
        #         "max_iter": 1000,
        #         "sb": "yes",
        #         "print_level": 0,
        #     },
        # }
        opts = {
            "print_time": False,
            "record_time": True,
            "error_on_fail": False,
            "printLevel": "none",
            "jit": False,
        }
        self.init_solver(opts, solver="qpoases")
