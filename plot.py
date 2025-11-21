import pickle

import matplotlib.pyplot as plt
import numpy as np

from tikz import save2tikz

skip_points = 5

cent_data = []
dist_data = []
dist_fo_data = []

for seed in [1, 2, 3, 4, 5]:
    with open(f"results/dist_so/dist_so_seed_{seed}_results_2000.pkl", "rb") as f:
        dist_data.append(pickle.load(f))
    with open(f"results/cent_so/cent_so_seed_{seed}_results_2000.pkl", "rb") as f:
        cent_data.append(pickle.load(f))
    with open(f"results/dist_fo/dist_fo_seed_{seed}_results_2000.pkl", "rb") as f:
        dist_fo_data.append(pickle.load(f))

TD = [
    np.vstack([np.abs(data["td"]) for data in dist_data]),
    np.vstack([np.abs(data["td"]) for data in cent_data]),
    np.vstack([np.abs(data["td"]) for data in dist_fo_data]),
]
L = [
    np.vstack([data["R"] for data in dist_data]),
    np.vstack([data["R"] for data in cent_data]),
    np.vstack([data["R"] for data in dist_fo_data]),
]
mvavg_window = 100
TD_mean = [np.median(td, axis=0) for td in TD]
TD_std = [np.std(td, axis=0) for td in TD]
TD_lower = [np.percentile(td, 32, axis=0) for td in TD]
TD_upper = [np.percentile(td, 68, axis=0) for td in TD]
L_mean = [np.median(l, axis=0) for l in L]
L_std = [np.std(l, axis=0) for l in L]
L_lower = [np.percentile(l, 32, axis=0) for l in L]
L_upper = [np.percentile(l, 68, axis=0) for l in L]
TD_mean_mvavg = [
    np.convolve(td, np.ones(mvavg_window) / mvavg_window, mode="valid")
    for td in TD_mean
]
TD_std_mvavg = [
    np.convolve(td, np.ones(mvavg_window) / mvavg_window, mode="valid") for td in TD_std
]
TD_lower_mvavg = [
    np.convolve(td, np.ones(mvavg_window) / mvavg_window, mode="valid")
    for td in TD_lower
]
TD_upper_mvavg = [
    np.convolve(td, np.ones(mvavg_window) / mvavg_window, mode="valid")
    for td in TD_upper
]
L_mean_mvavg = [
    np.convolve(l, np.ones(mvavg_window) / mvavg_window, mode="valid") for l in L_mean
]
L_std_mvavg = [
    np.convolve(l, np.ones(mvavg_window) / mvavg_window, mode="valid") for l in L_std
]
L_lower_mvavg = [
    np.convolve(l, np.ones(mvavg_window) / mvavg_window, mode="valid") for l in L_lower
]
L_upper_mvavg = [
    np.convolve(l, np.ones(mvavg_window) / mvavg_window, mode="valid") for l in L_upper
]

colors = ["C0", "C1", "C2"]
labels = ["D-SO", "C-SO", "D-FO"]
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
for i in range(len(TD_mean_mvavg)):
    axs[0].plot(
        np.arange(len(TD_mean_mvavg[i]))[::skip_points],
        TD_mean_mvavg[i][::skip_points],
        color=colors[i],
        label=labels[i],
    )
    axs[0].fill_between(
        np.arange(len(TD_mean_mvavg[i]))[::skip_points],
        # TD_mean_mvavg[i] - TD_std_mvavg[i],
        TD_lower_mvavg[i][::skip_points],
        # TD_mean_mvavg[i] + TD_std_mvavg[i],
        TD_upper_mvavg[i][::skip_points],
        color=colors[i],
        alpha=0.3,
    )
    axs[0].set_ylabel("TD")

    axs[1].plot(
        np.arange(len(L_mean_mvavg[i]))[::skip_points],
        L_mean_mvavg[i][::skip_points],
        color=colors[i],
    )
    axs[1].fill_between(
        np.arange(len(L_mean_mvavg[i]))[::skip_points],
        # np.clip(L_mean_mvavg[i] - L_std_mvavg[i], 1e-2, None),
        L_lower_mvavg[i][::skip_points],
        # L_mean_mvavg[i] + L_std_mvavg[i],
        L_upper_mvavg[i][::skip_points],
        color=colors[i],
        alpha=0.3,
    )
    axs[1].set_ylabel("$L$")
axs[0].legend()
axs[1].set_xlabel("k")
axs[1].set_yscale("log")
save2tikz(plt.gcf())

fig, axs = plt.subplots(3, 3, constrained_layout=True, sharex=True)
item = 0
for i, data in enumerate([dist_data[item], cent_data[item], dist_fo_data[item]]):
    X = data["X"]
    if i == 1:
        X = X.T
    U = data["U"]

    for k, j in enumerate([0, 2, 4]):
        axs[0, i].plot(
            np.arange(np.max(X.shape))[::skip_points],
            X[:, j].squeeze()[::skip_points],
            label=[f"x1 agent {k + 1}"],
            color=f"C{k}",
        )
    axs[0, i].axhline(0, color="red", linestyle="--", label="x1 lower bound")
    axs[0, i].axhline(1, color="red", linestyle="--", label="x1 upper bound")

    for k, j in enumerate([1, 3, 5]):
        axs[1, i].plot(
            np.arange(np.max(X.shape))[::skip_points],
            X[:, j].squeeze()[::skip_points],
            label=[f"x2 agent {k + 1}"],
            color=f"C{k}",
        )
    axs[1, i].axhline(-1, color="red", linestyle="--", label="x2 lower bound")
    axs[1, i].axhline(1, color="red", linestyle="--", label="x2 upper bound")

    for k in range(U.shape[1]):
        axs[2, i].plot(
            np.arange(U.shape[0])[::skip_points],
            U[:, k].squeeze()[::skip_points],
            label=[f"u agent {k + 1}"],
            color=f"C{k}",
        )
    axs[2, i].axhline(-1, color="red", linestyle="--", label="u lower bound")
    axs[2, i].axhline(1, color="red", linestyle="--", label="u upper bound")
    axs[2, i].set_xlabel("k")
axs[0, 0].set_ylabel("x1")
axs[1, 0].set_ylabel("x2")
axs[2, 0].set_ylabel("u")

axs[0, 0].set_title("D-SO")
axs[0, 1].set_title("C-SO")
axs[0, 2].set_title("D-FO")
save2tikz(plt.gcf())

# fig, axs = plt.subplots(len(data_1["updates"]), 1, constrained_layout=True, sharex=True)
# sorted_keys = [
#     sorted(
#         updates.keys(),
#         key=lambda k: np.linalg.norm(updates[k][-1] - updates[k][0]),
#         reverse=True,
#     )
#     for updates in data_1["updates"]
# ]
# for i, updates in enumerate(data_1["updates"]):
#     axs[i].plot(np.asarray(updates[sorted_keys[i][0]]).squeeze(), color="C0")
#     axs[i].plot(
#         np.asarray(data_2["updates"][f"{sorted_keys[i][0]}_{i}"]).squeeze(),
#         linestyle="--",
#         color="C1",
#     )
#     axs[i].set_ylabel(f"{sorted_keys[i][0]}_{i}")

# axs[-1].set_xlabel("k")
# save2tikz(plt.gcf())
plt.show()
