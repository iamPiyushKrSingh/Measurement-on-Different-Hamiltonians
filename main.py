#!/home/piyush/miniconda3/envs/jupyter/bin/python

from Free_Fermions_Model.hamiltonians import anderson_hamiltonian
from Free_Fermions_Model.spacetime_distribution import (
    spacetime_probability_distribution_finer_disorder,
)


import numpy as np
import jax.numpy as jnp

from jax import config

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scienceplots

plt.style.use(["science", "ieee"])


def main():
    L = 101
    J = 1.0
    W = 2
    seed = 694269

    H = anderson_hamiltonian(L, J, W, seed, False)

    Δt = 10
    n = 100
    generations = 300

    psi = jnp.zeros(L).at[(L - 1) // 2].set(1)

    prob_new = spacetime_probability_distribution_finer_disorder(
        H, psi, Δt, n, generations
    )

    fig, ax = plt.subplots()

    X = np.arange(0, (generations * (n + 1) + 1) / 100, 0.01)
    Y = np.arange(0, L, 1)

    pc = ax.pcolormesh(
        X,
        Y,
        np.where(prob_new <= 1e-30, 1e-30, prob_new).T,
        cmap="viridis",
        norm=colors.LogNorm(vmin=1e-5, vmax=1),
        shading="auto",
    )
    fig.colorbar(pc, ax=ax, extend="min")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Site")
    ax.set_title("Spacetime Probability Distribution for Anderson Localization")
    plt.tight_layout()
    plt.savefig("check.png", dpi=300, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    main()
