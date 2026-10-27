from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.ns2d import NavierStokes2D


def initial_vorticity(solver):
    """Same initial condition for all simulations."""
    x, y = solver.X, solver.Y

    return (
        2 * np.sin(x) * np.sin(y)
        + 0.6 * np.cos(3 * x + 2 * y)
        + 0.4 * np.sin(2 * x - 3 * y)
    )


def main():
    viscosities = [0.05, 0.01, 0.005]

    N = 64
    dt = 0.005
    steps = 200

    output = Path(__file__).resolve().parent.parent / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8), constrained_layout=True
    )

    for i, nu in enumerate(viscosities):
        solver = NavierStokes2D(N=N, nu=nu)

        solver.set_initial_vorticity(
            initial_vorticity(solver)
        )

        for _ in range(steps):
            solver.step(dt)

        u, v = solver.velocity()
        speed = np.sqrt(u**2 + v**2)
        omega = solver.vorticity()

        axes[0, i].imshow(
            speed.T,
            origin="lower",
            extent=[0, solver.L, 0, solver.L],
            vmin=0,
            vmax=1.3
        )
        axes[0, i].set_title(f"Velocity, nu={nu}")

        axes[1, i].imshow(
            omega.T,
            origin="lower",
            extent=[0, solver.L, 0, solver.L],
            cmap="RdBu_r",
            vmin=-3,
            vmax=3
        )
        axes[1, i].set_title(f"Vorticity, nu={nu}")

        print(
            f"nu={nu}, "
            f"energy={solver.kinetic_energy():.6f}"
        )

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.savefig(
        output / "viscosity_comparison.png",
        dpi=160
    )

    plt.show()


if __name__ == "__main__":
    main()
