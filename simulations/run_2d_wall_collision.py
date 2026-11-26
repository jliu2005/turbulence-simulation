"""Film a vortex dipole approaching two stationary, impermeable top walls."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize

from src.ns2d_walls import NavierStokes2DWalls, Wall


def initial_vorticity(solver):
    """Apply the solver's discrete negative Laplacian to a compact dipole.

    Opposite streamfunction bumps on the left and right drive the pair upward.
    The initial streamfunction vanishes in a neighborhood of every wall.
    """
    psi = np.zeros_like(solver.X)
    for center_x, sign in ((0.44, 1), (0.56, -1)):
        r_squared = (solver.X - center_x) ** 2 + (solver.Y - 0.52) ** 2
        psi += sign * 0.17 * np.maximum(1 - r_squared / 0.19**2, 0.0) ** 8
    omega = np.zeros_like(psi)
    omega[1:-1, 1:-1] = (
        4 * psi[1:-1, 1:-1]
        - psi[:-2, 1:-1]
        - psi[2:, 1:-1]
        - psi[1:-1, :-2]
        - psi[1:-1, 2:]
    ) / solver.h**2
    return omega


def matched_pair(n):
    """Construct the two unit-square runs in free-slip, no-slip order."""
    walls = dict.fromkeys(("bottom", "left", "right", "top"), Wall("free_slip"))
    free = NavierStokes2DWalls(N=n, L=1, nu=0.001, walls=walls)
    rigid = NavierStokes2DWalls(N=n, L=1, nu=free.nu, walls={**walls, "top": Wall()})
    omega = initial_vorticity(free)
    free.set_initial_vorticity(omega)
    rigid.set_initial_vorticity(omega)
    return free, rigid


def sampled_runs(n):
    """Yield the two live solver states every 0.01 until time 1.20."""
    free, rigid = matched_pair(n)
    u, v = free.velocity()
    speed = np.hypot(u, v).max()
    interval = 0.01
    max_dt = min(0.45 * free.h / speed, 0.18 * free.h**2 / free.nu)
    steps = int(np.ceil(interval / max_dt))
    dt = interval / steps
    yield 0.0, free, rigid
    for frame in range(1, 121):
        for _ in range(steps):
            free.step(dt)
            rigid.step(dt)
        yield frame * interval, free, rigid


def snapshots(n):
    """Yield synchronized vorticity snapshots for the movie."""
    for t, free, rigid in sampled_runs(n):
        yield t, free.vorticity(), rigid.vorticity()


def save_movie(n, output):
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Rendering MP4 requires ffmpeg on PATH.")
    data = snapshots(n)
    _, free, rigid = next(data)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.82, wspace=0.09)
    fig.suptitle("Vortex pair approaching the upper wall", fontsize=17, y=0.98)
    norm = Normalize(vmin=-75, vmax=75)
    images = []
    h = 1 / (n - 1)
    extent = [-h / 2, 1 + h / 2, -h / 2, 1 + h / 2]
    for ax, field, label in zip(
        axes, (free, rigid), ("Free-slip wall", "No-slip wall")
    ):
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            cmap="RdBu_r",
            norm=norm,
            aspect="equal",
        )
        images.append(im)
        ax.set(xlim=(0, 1), ylim=(0.26, 1), xlabel="x / L")
        ax.set_title(label, fontsize=12)
        ax.spines["top"].set_linewidth(2.0)
        ax.set_yticks([0.4, 0.6, 0.8, 1])
    axes[0].set_ylabel("y / L")
    axes[1].tick_params(labelleft=False)
    bar_ax = fig.add_axes([0.29, 0.12, 0.42, 0.027])
    bar = fig.colorbar(
        images[0],
        cax=bar_ax,
        orientation="horizontal",
        extend="both",
        ticks=[-75, -50, -25, 0, 25, 50, 75],
    )
    bar.set_label("Signed vorticity   (blue: clockwise, red: counterclockwise)")
    clock = fig.text(0.5, 0.89, "t = 0.00", ha="center", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=12,
        codec="libx264",
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "21",
            "-movflags",
            "+faststart",
        ],
    )
    # Finish the encoder before replacing a previous movie on a rerun.
    try:
        with TemporaryDirectory(dir=output.parent) as directory:
            temp = Path(directory) / output.name
            with writer.saving(fig, temp, dpi=100):
                writer.grab_frame()
                for t, free, rigid in data:
                    images[0].set_data(free.T)
                    images[1].set_data(rigid.T)
                    clock.set_text(f"t = {t:.2f}")
                    writer.grab_frame()
            temp.replace(output)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        type=int,
        default=193,
        help="nodes per side, including the walls (default: 193)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/walls"))
    args = parser.parse_args()
    if args.nodes < 65:
        parser.error(
            "use at least 65 nodes per side to sample the initial vortex cores"
        )
    output = args.output_dir / "vortex_wall_collision.mp4"
    save_movie(args.nodes, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
