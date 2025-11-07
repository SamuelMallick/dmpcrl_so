import pickle

import matplotlib.pyplot as plt
import numpy as np

# from tikz import save2tikz

with open("cent_results_900.pkl", "rb") as f:
    data_1 = pickle.load(f)
with open("cent_results_900.pkl", "rb") as f:
    data_2 = pickle.load(f)

colors = ["C1", "C0"]
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for i, data in enumerate([data_2, data_1]):
    R = data["R"]
    td = data["td"]

    axs[0].plot(
        td[:],
        color=colors[i],
        marker="o",
        linestyle="None",
        markersize=1,
        label=["Centralized", "Distributed"][i],
    )
    axs[0].set_ylabel("TD")
    axs[1].plot(R[:], color=colors[i], marker="o", linestyle="None", markersize=1)
    axs[1].set_ylabel("$L$")
    axs[2].plot(R[:], color=colors[i], marker="o", linestyle="None", markersize=1)
    axs[2].set_ylabel("$L$ log")
axs[0].legend()
axs[2].set_xlabel("k")
axs[2].set_yscale("log")
# save2tikz(plt.gcf())

fig, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
for i, data in enumerate([data_1, data_2]):
    X = data["X"]
    if i == 1:
        X = X.T
    U = data["U"]

    for k, j in enumerate([0, 2, 4]):
        axs[0, i].plot(X[:, j].squeeze(), label=[f"x1 agent {k + 1}"], color=f"C{k}")
    axs[0, i].legend()
    axs[0, i].axhline(0, color="red", linestyle="--", label="x1 lower bound")
    axs[0, i].axhline(1, color="red", linestyle="--", label="x1 upper bound")

    for k, j in enumerate([1, 3, 5]):
        axs[1, i].plot(X[:, j].squeeze(), label=[f"x2 agent {k + 1}"], color=f"C{k}")
    axs[1, i].axhline(-1, color="red", linestyle="--", label="x2 lower bound")
    axs[1, i].axhline(1, color="red", linestyle="--", label="x2 upper bound")

    for k in range(U.shape[1]):
        axs[2, i].plot(U[:, k].squeeze(), label=[f"u agent {k + 1}"], color=f"C{k}")
    axs[2, i].axhline(-1, color="red", linestyle="--", label="u lower bound")
    axs[2, i].axhline(1, color="red", linestyle="--", label="u upper bound")
axs[0, 0].set_ylabel("x1")
axs[1, 0].set_ylabel("x2")
axs[2, 0].set_ylabel("u")
axs[2, 0].set_xlabel("k")
axs[2, 1].set_xlabel("k")
axs[0, 0].set_title("Distributed")
axs[0, 1].set_title("Centralized")
# save2tikz(plt.gcf())

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
