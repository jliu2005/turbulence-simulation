"""Verify the impulsively started unit lid-driven cavity at Reynolds number 100."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from src.ns2d_walls import NavierStokes2DWalls, Wall

# Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387–411, Tables I and II.
# https://doi.org/10.1016/0021-9991(82)90058-4
# Coordinates use the authors' 129-node grid-point indices to avoid rounding.
Y_GHIA = (
    np.array([0, 7, 8, 9, 13, 22, 36, 58, 64, 79, 94, 109, 122, 123, 124, 125, 128])
    / 128
)
U_GHIA = np.array(
    [
        0.0,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.0,
    ]
)
X_GHIA = (
    np.array([0, 8, 9, 10, 12, 20, 29, 30, 64, 103, 110, 116, 121, 122, 123, 124, 128])
    / 128
)
V_GHIA = np.array(
    [
        0.0,
        0.09233,
        0.10091,
        0.10890,
        0.12317,
        0.16077,
        0.17507,
        0.17527,
        0.05454,
        -0.24533,
        -0.22445,
        -0.16914,
        -0.10313,
        -0.08864,
        -0.07391,
        -0.05906,
        0.0,
    ]
)

SAMPLE_INTERVAL = 0.25  # nondimensional time L / U_lid
STEADY_TOL = 1e-6  # maximum velocity component change per unit time


@dataclass
class CavityRun:
    """Computed centerline profiles, steady-state history, and optional movie frames."""

    n: int
    u_center: np.ndarray
    v_center: np.ndarray
    changes: list
    startup_velocity: np.ndarray
    frames: list

    def max_profile_errors(self):
        grid = np.linspace(0, 1, self.n)
        du = np.interp(Y_GHIA, grid, self.u_center) - U_GHIA
        dv = np.interp(X_GHIA, grid, self.v_center) - V_GHIA
        return np.max(np.abs(du)), np.max(np.abs(dv))


def run(n, steps_per_sample, *, movie=False):
    """Advance from rest to a steady state, recording full-field velocity changes."""
    solver = NavierStokes2DWalls(N=n, L=1, nu=0.01, walls={"top": Wall("no_slip", 1.0)})
    dt = SAMPLE_INTERVAL / steps_per_sample
    previous = np.array(solver.velocity())
    frames = [(0.0, solver.vorticity(), solver.streamfunction())] if movie else []
    changes = []
    for sample in range(1, 161):  # allow up to 40 lid travel times
        for _ in range(steps_per_sample):
            solver.step(dt)
        t = sample * SAMPLE_INTERVAL
        velocity = np.array(solver.velocity())
        change = np.max(np.abs(velocity - previous)) / SAMPLE_INTERVAL
        if not np.isfinite(change):
            raise RuntimeError(f"N={n} produced nonfinite velocities at t={t:g}")
        changes.append(change)
        if t == 1.0:
            startup_velocity = velocity.copy()
        if movie and t <= 8:
            frames.append((t, solver.vorticity(), solver.streamfunction()))
        if t >= 1.0 and change < STEADY_TOL:
            mid = n // 2
            return CavityRun(
                n,
                velocity[0, mid],
                velocity[1, :, mid],
                changes,
                startup_velocity,
                frames,
            )
        previous = velocity
    raise RuntimeError(
        f"N={n} did not reach the steady criterion by t={t:g}: {change:g}"
    )


def plot_profiles(runs, output):
    fig, (ax_u, ax_v, ax_rate) = plt.subplots(
        1, 3, figsize=(12, 4), layout="constrained"
    )
    for result, color, style in zip(runs, ["#29616f", "#a34416"], ["--", "-"]):
        grid = np.linspace(0, 1, result.n)
        label = f"{result.n} × {result.n}"
        ax_u.plot(result.u_center, grid, style, color=color, label=label)
        ax_v.plot(grid, result.v_center, style, color=color)
        times = np.arange(1, len(result.changes) + 1) * SAMPLE_INTERVAL
        ax_rate.semilogy(times, result.changes, style, color=color)
    markers = dict(
        marker="o", linestyle="none", markerfacecolor="none", color="0.2", markersize=5
    )
    ax_u.plot(U_GHIA, Y_GHIA, label="Ghia et al.", **markers)
    ax_v.plot(X_GHIA, V_GHIA, **markers)
    ax_u.set(
        title="Vertical centerline", xlabel="Horizontal velocity  u / U", ylabel="y / L"
    )
    ax_v.set(
        title="Horizontal centerline", xlabel="x / L", ylabel="Vertical velocity  v / U"
    )
    ax_u.legend(frameon=False)
    ax_rate.axhline(STEADY_TOL, color="0.4", lw=1, ls=":", label="Stopping criterion")
    ax_rate.set(
        title="Approach to steady flow",
        xlabel="Time  tU / L",
        ylabel="Max. velocity change per unit time",
    )
    ax_rate.legend(frameon=False)
    for ax in (ax_u, ax_v, ax_rate):
        ax.grid(alpha=0.2)
    fig.suptitle("Lid-driven cavity   ·   Re = 100")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_movie(result, output):
    """Draw signed vorticity with contours of the streamfunction at equal times."""
    fig, ax = plt.subplots(figsize=(5.4, 4.4), layout="constrained")
    grid = np.linspace(0, 1, result.n)
    colors = ax.pcolormesh(
        grid,
        grid,
        result.frames[0][1].T,
        shading="nearest",
        cmap="RdBu_r",
        vmin=-8,
        vmax=8,
    )
    fig.colorbar(colors, ax=ax, label="Vorticity  ωL / U", extend="both")
    ax.set(xlabel="x / L", ylabel="y / L", aspect="equal", xlim=(0, 1), ylim=(0, 1))
    title = ax.set_title("")
    levels = [-0.09, -0.07, -0.05, -0.03, -0.01, -0.003]
    writer = PillowWriter(fps=6)
    with writer.saving(fig, output, dpi=70):
        for t, omega, psi in result.frames:
            colors.set_array(omega.T)
            title.set_text(f"Right-moving lid   ·   Re = 100   ·   tU/L = {t:.2f}")
            active = [level for level in levels if psi.min() < level < psi.max()]
            contours = None
            if active:
                contours = ax.contour(
                    grid,
                    grid,
                    psi.T,
                    levels=active,
                    colors="0.2",
                    linewidths=0.65,
                    linestyles="solid",
                )
            writer.grab_frame()
            if contours is not None:
                contours.remove()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grids",
        type=int,
        nargs=2,
        default=(33, 65),
        metavar=("COARSE", "FINE"),
        help="odd endpoint-inclusive node counts",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/walls"))
    parser.add_argument("--no-movie", action="store_true")
    args = parser.parse_args()
    coarse, fine = args.grids
    if not (5 <= coarse < fine and coarse % 2 and fine % 2):
        parser.error("choose two increasing odd grids with at least five nodes")
    h = 1 / (fine - 1)
    # Shared step for both grids. Respect advection and diffusion restrictions.
    limit = min(0.4 * h, 0.18 * h**2 / 0.01)
    steps = int(np.ceil(SAMPLE_INTERVAL / limit))
    dt = SAMPLE_INTERVAL / steps
    coarse_run = run(coarse, steps)
    fine_run = run(fine, steps, movie=not args.no_movie)
    for result in (coarse_run, fine_run):
        err_u, err_v = result.max_profile_errors()
        print(
            f"N={result.n}, dt={dt:.8g}, t={len(result.changes) * SAMPLE_INTERVAL:g}, "
            f"steady rate={result.changes[-1]:.3g}, max Ghia |du|={err_u:.6f}, "
            f"max Ghia |dv|={err_v:.6f}"
        )
    # Test the time step during startup, before the steady state erases its error.
    half = NavierStokes2DWalls(
        N=fine, L=1, nu=0.01, walls={"top": Wall("no_slip", 1.0)}
    )
    for _ in range(8 * steps):  # t = 1 at dt / 2
        half.step(dt / 2)
    delta = np.max(np.abs(np.array(half.velocity()) - fine_run.startup_velocity))
    print(f"N={fine}, t=1, max velocity change on halving dt={delta:.6g}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_profiles([coarse_run, fine_run], args.output_dir / "cavity_profiles.png")
    if not args.no_movie:
        save_movie(fine_run, args.output_dir / "cavity_startup.gif")


if __name__ == "__main__":
    main()
