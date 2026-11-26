"""Check a decaying vortex pair against the exact free-slip solution."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.ns2d_walls import NavierStokes2DWalls, Wall


def evolve_mode(n, times, nu):
    """Return vorticity snapshots and their amplitudes on the unit square."""
    walls = {side: Wall("free_slip") for side in ("left", "right", "bottom", "top")}
    solver = NavierStokes2DWalls(N=n, L=1.0, nu=nu, walls=walls)
    initial = np.sin(2 * np.pi * solver.X) * np.sin(np.pi * solver.Y)
    solver.set_initial_vorticity(initial)

    snapshots = [solver.vorticity()]
    for interval in np.diff(times):
        # The diffusion limit decreases with the square of the grid spacing.
        steps = int(np.ceil(interval / (0.20 * solver.h**2 / nu)))
        for _ in range(steps):
            solver.step(interval / steps)
        snapshots.append(solver.vorticity())

    snapshots = np.stack(snapshots)
    amplitudes = np.sum(snapshots * initial, axis=(1, 2)) / np.sum(initial**2)
    exact_final = np.exp(-5 * np.pi**2 * nu * times[-1]) * initial
    error = np.linalg.norm(snapshots[-1] - exact_final) / np.linalg.norm(exact_final)
    return snapshots, amplitudes, error


def save_movie(path, times, snapshots, exact_amplitudes):
    """Animate the computed field and overlay its midline with the exact solution."""
    fig, (field_ax, line_ax) = plt.subplots(
        1, 2, figsize=(7.5, 3.4), layout="constrained"
    )
    n = snapshots.shape[1]
    x = np.linspace(0, 1, n)
    levels = dict(vmin=-1, vmax=1, cmap="RdBu_r")
    field = field_ax.pcolormesh(x, x, snapshots[0].T, shading="gouraud", **levels)
    field_ax.set(xlabel="x", ylabel="y", aspect="equal")
    clock = field_ax.set_title("Vorticity at t = 0.00")
    fig.colorbar(field, ax=field_ax, shrink=0.85)
    (exact,) = line_ax.plot(x, snapshots[0, :, n // 2], color="black", label="Exact")
    (measured,) = line_ax.plot(
        x[::4],
        snapshots[0, ::4, n // 2],
        "o",
        ms=4,
        mfc="white",
        color="#21618c",
        label="Solver",
    )
    line_ax.set(
        xlim=(0, 1),
        ylim=(-1.05, 1.05),
        xlabel="x",
        ylabel="Vorticity",
        title="Slice at y = 0.5",
    )
    line_ax.legend(loc="upper right")
    line_ax.axhline(0, color="0.8", lw=0.7, zorder=0)

    def update(i):
        field.set_array(snapshots[i].T)
        clock.set_text(f"Vorticity at t = {times[i]:.2f}")
        measured.set_ydata(snapshots[i, ::4, n // 2])
        exact.set_ydata(np.sin(2 * np.pi * x) * exact_amplitudes[i])

    movie = FuncAnimation(fig, update, frames=len(times), repeat=False)
    movie.save(path, writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/walls"))
    parser.add_argument("--no-movie", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nu = 0.025
    times = np.linspace(0, 1, 25)
    exact = np.exp(-5 * np.pi**2 * nu * times)
    grids = (17, 33, 65)
    errors = []
    fig, (decay, refinement) = plt.subplots(
        1, 2, figsize=(9, 3.8), layout="constrained"
    )
    decay.plot(times, exact, "k-", label="Exact")
    for n, marker in zip(grids, ("+", "x", "o"), strict=True):
        snapshots, amplitudes, error = evolve_mode(n, times, nu)
        errors.append(error)
        print(f"N={n:2d}  relative vorticity error at t=1: {error:.6e}")
        decay.plot(
            times[::4],
            amplitudes[::4],
            marker,
            ms=5,
            fillstyle="none",
            label=f"{n} × {n}",
        )
    h = 1 / (np.array(grids) - 1)
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    print("Observed orders:", ", ".join(f"{order:.3f}" for order in orders))
    decay.set(xlabel="Time", ylabel="Mode amplitude", title="Free-slip viscous decay")
    decay.legend()
    refinement.loglog(h, errors, "o-", label="Measured error")
    refinement.loglog(
        h, 0.72 * errors[-1] * (h / h[-1]) ** 2, "k--", label="Second order"
    )
    refinement.set(
        xlabel="Grid spacing",
        ylabel="Relative field error at t = 1",
        title="Spatial refinement",
    )
    refinement.legend()
    fig.savefig(args.output_dir / "free_slip_decay.png", dpi=145)
    plt.close(fig)
    if not args.no_movie:
        save_movie(args.output_dir / "free_slip_decay.gif", times, snapshots, exact)


if __name__ == "__main__":
    main()
