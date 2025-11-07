import itertools
import logging
import pickle
from typing import Optional

import numpy as np
from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from mpcrl import LstdQLearningAgent

from optimizers import DistributedSecondOrderOptimizer


class LocalAgent(LstdQLearningAgent):
    def _try_store_experience(self, cost, solQ, solV):
        # overriding the way experience is stored for second order optimizers
        # to save also quantities needed for the distributed second order update
        success = solQ.success and solV.success
        if success:
            td_error = cost + self.discount_factor * solV.f - solQ.f
            if self.optimizer.order == 1:
                dQ = self._sensitivity(solQ)
                gradient = -td_error * dQ
                self.store_experience((gradient, None, dQ, None, td_error))
            else:
                dQ, ddQ = self._sensitivity(solQ)
                gradient = -td_error * dQ
                hessian = np.multiply.outer(dQ, dQ) - td_error * ddQ
                h = td_error * ddQ
                self.store_experience((gradient, hessian, dQ, h, td_error))
        else:
            td_error = np.nan

        if self.td_errors is not None:
            self.td_errors.append(td_error)
        return success

    def update(self) -> Optional[str]:

        # this self.sample has been set by the coordinator at the timestep end,
        # after the sample was also used for the consensus step
        sample = self.sample
        if self.optimizer.order == 1:
            gradients, _, _, _, _ = zip(*sample)
            gradient = np.mean(list(gradients), 0)
            return self.optimizer.update(gradient)

        if not isinstance(self.optimizer, DistributedSecondOrderOptimizer):
            raise TypeError(
                f"`{self.__class__.__name__}` agent requires a "
                f"`{DistributedSecondOrderOptimizer.__name__}` optimizer when "
                f"`hessian_type` is not 'none', but got "
                f"`{self.optimizer.__class__.__name__}`."
            )
        gradients, hessians, dQs, hs, td_errors = zip(*sample)
        gradient = np.mean(gradients, 0)
        hessian = np.mean(hessians, 0)
        return self.optimizer.update(
            gradient,
            hessian,
            {
                "gradients": gradients,
                "hessians": hessians,
                "dQs": dQs,
                "hs": hs,
                "td_errors": td_errors,
            },
        )


class Coordinator(LstdQLearningAgentCoordinator):
    save_frequency = 0

    def on_timestep_end(self, env, episode, timestep):

        # saving
        if self.save_frequency > 0 and timestep % self.save_frequency == 0:
            X = np.asarray(env.get_wrapper_attr("ep_observations"))
            U = np.asarray(env.get_wrapper_attr("ep_actions"))
            R = np.asarray(env.get_wrapper_attr("ep_rewards"))
            td = self.td_errors if self.centralized_flag else self.agents[0].td_errors
            with open(f"{self.save_name}_results_{timestep}.pkl", "wb") as f:
                pickle.dump(
                    {"X": X, "U": U, "R": R, "td": td},
                    f,
                )
                logging.info(f"Results saved to results for timestep {timestep}.pkl")

        # handling consensus of latest experience of local agents
        samples = []
        for i in range(len(self.agents)):
            sample = list(self.agents[i].experience.sample())
            self.agents[i].unwrapped.sample = sample.copy()
            samples.append(sample)
        if not self.centralized_flag and not self.agents[0].optimizer.order == 1:
            if not all(len(s) == len(samples[0]) for s in samples):
                raise ValueError(
                    "All agents must have the same number of samples to perform consensus."
                )
            T = len(samples[0])
            gss = []
            K_tilde_is = []
            for i in range(len(self.agents)):
                _, _, gs, hs, _ = zip(*samples[i])
                K_i = np.mean(hs, 0)
                if (
                    K_i.shape != ()
                ):  # if we have matrix indicating second order derivatives present
                    tau = 1e-3  # tau is added here for non-singularity, which is then checked below
                    sigma = tau / T
                    self.agents[i].optimizer.sigma = sigma

                    # an error will be raised here, from numpy.linalg.inv, if the matrix is singular
                    K_tilde_is.append(
                        np.linalg.inv(T * sigma * np.eye(gs[0].shape[0]) - K_i)
                    )
                else:
                    tau = 1e-3
                    sigma = tau / T
                    self.agents[i].optimizer.sigma = sigma
                    K_tilde_is.append(np.linalg.inv(T * sigma * np.eye(gs[0].shape[0])))
                gss.append(np.array(gs).T)

            # consensus used to generate C
            C_bar = [
                np.array(
                    [
                        gss[i][:, k1].T @ K_tilde_is[i] @ gss[i][:, k2]
                        for k1 in range(T)
                        for k2 in range(k1, T)
                    ]
                )
                for i in range(len(gss))
            ]
            C_consensus_vec = np.array(C_bar)
            C_bar = np.split(
                self.consensus_coordinator.average_consensus(self.n * C_consensus_vec),
                self.n,
            )
            for i in range(self.n):
                C = np.zeros((T, T))
                C_bar_ = C_bar[i].reshape(
                    -1,
                )
                k = 0
                for k1 in range(T):
                    for k2 in range(k1, T):
                        C[k1, k2] = C_bar_[k]
                        k += 1
                C += C.T - np.diag(C.diagonal())

                # C then set locally for each agent, to mimic local consensus
                self.agents[i].optimizer.C = C
        return super().on_timestep_end(env, episode, timestep)

    def train(
        self,
        env,
        episodes,
        seed=None,
        raises=True,
        env_reset_options=None,
        save_frequency=0,
        save_name="",
    ):
        self.save_frequency = save_frequency
        self.save_name = save_name
        return super().train(env, episodes, seed, raises, env_reset_options)
