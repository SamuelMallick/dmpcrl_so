import logging
import pickle

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from csnlp.wrappers import Mpc
from dmpcrl.core.admm import AdmmCoordinator
from gymnasium.wrappers import TimeLimit
from mpcrl import (
    ExperienceReplay,
    LearnableParameter,
    LearnableParametersDict,
    UpdateStrategy,
)
from mpcrl.core.exploration import EpsilonGreedyExploration, StepWiseExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.optim import GradientDescent
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from agent import Coordinator, LocalAgent
from env import LtiNetwork
from mpc import GlobalLearningMpc, LocalLearningMpc
from optimizers import CentralizedSecondOrderOptimizer, DistributedSecondOrderOptimizer

cent_flag = False
second_order = True

plot = True
save = True
sim_len = 2001

n = 3
N = 10
gamma = 0.9
Adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
G = AdmmCoordinator.g_map(Adj)  # mapping from global var to local var indexes for ADMM
env = MonitorEpisodes(TimeLimit(LtiNetwork(n, Adj), max_episode_steps=int(sim_len)))

rho = 1
admm_iters = 100
consensus_iters = 100

# cent mpc
mpc = GlobalLearningMpc(N, n, Adj, gamma)
cent_learnable_parameters = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val)
        for name, val in mpc.learnable_pars_init.items()
    )
)


# distributed mpc and params
mpcs: list[Mpc] = []
learnable_parameters: list[LearnableParametersDict] = []
fixed_parameters: list = []
for i in range(n):
    mpcs.append(
        LocalLearningMpc(
            prediction_horizon=N,
            num_neighbours=len(G[i]) - 1,
            my_index=G[i].index(i),
            gamma=gamma,
            rho=rho,
        )
    )
    learnable_parameters.append(
        LearnableParametersDict[cs.SX](
            (
                LearnableParameter(name, val.shape, val)
                for name, val in mpcs[i].learnable_pars_init.items()
            )
        )
    )
    fixed_parameters.append(mpcs[i].fixed_pars_init)
agents = [
    RecordUpdates(
        LocalAgent(
            mpcs[i],
            update_strategy=1,
            discount_factor=gamma,
            optimizer=(
                DistributedSecondOrderOptimizer(
                    learning_rate=ExponentialScheduler(1e-4, factor=1)
                )
                if second_order
                else GradientDescent(learning_rate=1e-8)
            ),
            learnable_parameters=learnable_parameters[i],
            fixed_parameters=fixed_parameters[i],
            hessian_type="approx" if second_order else "none",
            record_td_errors=True,
            experience=ExperienceReplay(
                maxlen=100, sample_size=15, include_latest=10, seed=1
            ),
            exploration=StepWiseExploration(
                EpsilonGreedyExploration(
                    epsilon=ExponentialScheduler(0, factor=0.997),
                    strength=0.5 * (2),
                    seed=1,
                    hook="on_timestep_end",
                ),
                step_size=admm_iters,
                stepwise_decay=False,
            ),
        )
    )
    for i in range(n)
]
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        Coordinator(
            agents=agents,
            N=N,
            nx=2,
            nu=1,
            adj=Adj,
            rho=rho,
            admm_iters=admm_iters,
            consensus_iters=consensus_iters,
            centralized_mpc=mpc,
            hessian_type="approx" if second_order else "none",
            centralized_flag=cent_flag,
            centralized_debug=False,
            centralized_learnable_parameters=cent_learnable_parameters,
            centralized_fixed_parameters={},
            centralized_update_strategy=UpdateStrategy(1, hook="on_timestep_end"),
            centralized_discount_factor=gamma,
            centralized_optimizer=(
                CentralizedSecondOrderOptimizer(
                    learning_rate=ExponentialScheduler(1e-4, factor=1)
                )
                if second_order
                else GradientDescent(learning_rate=1e-8)
            ),
            centralized_exploration=EpsilonGreedyExploration(
                epsilon=ExponentialScheduler(0, factor=0.99),
                strength=0.5 * (2),
                seed=1,
            ),
            centralized_experience=ExperienceReplay(
                maxlen=100, sample_size=15, include_latest=10, seed=1
            ),
            record_td_errors=True,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

agent.train(
    env=env, episodes=1, seed=5, save_frequency=1000, save_name="dist_fo_seed_5"
)
# agent.evaluate(env=env, episodes=1, seed=1)

# Plotting the results
X = env.get_wrapper_attr("observations")[0].squeeze().T
U = env.get_wrapper_attr("actions")[0].squeeze()
R = env.get_wrapper_attr("rewards")[0]
td = agent.td_errors if cent_flag else agents[0].td_errors

if save:
    with open("results.pkl", "wb") as f:
        pickle.dump(
            {"X": X, "U": U, "R": R, "td": td},
            f,
        )
    logging.info("Results saved to results.pkl")

if plot:
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(X[[0, 2, 4], :].T, label=["x1 agent 1", "x1 agent 2", "x1 agent 3"])
    axs[0].legend()
    axs[0].axhline(
        env.unwrapped.x_bnd[0, 0], color="red", linestyle="--", label="x1 lower bound"
    )
    axs[0].axhline(
        env.unwrapped.x_bnd[1, 0], color="green", linestyle="--", label="x1 upper bound"
    )
    axs[0].set_title("State x1")
    axs[1].plot(X[[1, 3, 5], :].T, label=["x2 agent 1", "x2 agent 2", "x2 agent 3"])
    axs[1].axhline(
        env.unwrapped.x_bnd[0, 1], color="red", linestyle="--", label="x2 lower bound"
    )
    axs[1].axhline(
        env.unwrapped.x_bnd[1, 1], color="green", linestyle="--", label="x2 upper bound"
    )
    axs[1].set_title("State x2")
    axs[2].plot(U, label="u")
    axs[2].axhline(
        env.unwrapped.u_bnd[0, 0], color="red", linestyle="--", label="u lower bound"
    )
    axs[2].axhline(
        env.unwrapped.u_bnd[1, 0], color="green", linestyle="--", label="u upper bound"
    )
    axs[2].set_title("Control input u")

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    x = np.arange(R.size)
    if td:
        axs[0].scatter(
            x,
            np.asarray(td),
            color="blue",
            marker="o",
        )
        axs[0].set_ylabel("td error")
    axs[1].scatter(x, R, color="blue", marker="o")
    axs[1].set_ylabel("$L$")

    plt.show()
