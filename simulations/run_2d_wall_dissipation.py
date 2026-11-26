"""Compare viscous losses in the top sixteenth of the vortex collision box."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from simulations.run_2d_wall_collision import sampled_runs


def dissipation_rates(density, h):
    """Integrate nodal dissipation over the whole square and its top strip."""
    intervals = density.shape[1] - 1
    if intervals % 16:
        raise ValueError("The strip boundary must coincide with a grid node")
    # Integrate in x first, then in y, with half weight at each region boundary.
    profile = trapezoid(density, dx=h, axis=0)
    strip = profile[-(intervals // 16 + 1) :]
    return trapezoid(profile, dx=h), trapezoid(strip, dx=h)


def compare(n):
    times, rates, energies = [], [], []
    for t, free, rigid in sampled_runs(n):
        times.append(t)
        rates.append(
            [dissipation_rates(s.dissipation_density(), s.h) for s in (free, rigid)]
        )
        energies.append([s.kinetic_energy() for s in (free, rigid)])
    return np.array(times), np.array(rates), np.array(energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/walls"))
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.27, top=0.77, wspace=0.34)
    handles = []
    for n in (129, 193):
        t, rates, energies = compare(n)
        for i, (name, color) in enumerate(
            (("Free slip", "#126987"), ("No slip", "#b04729"))
        ):
            total, strip = rates[:, i, :].T
            total_loss = trapezoid(total, x=t)
            strip_loss = cumulative_trapezoid(strip, x=t, initial=0)
            fraction = 100 * strip / total
            e0 = energies[0, i]  # unit box, hence mean energy equals integrated energy
            kw = dict(color=color, linewidth=1.9)
            if n == 129:
                kw.update(
                    linestyle="none",
                    marker="o",
                    markerfacecolor="none",
                    markersize=5.5,
                    markevery=4,
                    zorder=3,
                )
            (line,) = axes[0].plot(t, strip, label=f"{name}, {n}²", **kw)
            handles.append(line)
            axes[1].plot(t, fraction, **kw)
            axes[2].plot(t, 100 * strip_loss / e0, **kw)
            print(
                f"{name}, {n}²: strip loss={strip_loss[-1]:.6f}, total loss={total_loss:.6f}, "
                f"strip/total={100 * strip_loss[-1] / total_loss:.2f}%, "
                f"strip/K0={100 * strip_loss[-1] / e0:.2f}%, peak strip rate={strip.max():.5f} "
                f"at t={t[strip.argmax()]:.2f}, peak fraction={fraction.max():.2f}%, "
                f"energy closure residual/K0={100 * (total_loss + energies[-1, i] - e0) / e0:.3f}%"
            )
    # Section 6 measures closest approach of the 193² no-slip cores at t=0.59.
    impact_time = 0.59
    for ax in axes:
        marker = ax.axvline(
            impact_time, color="0.3", linestyle="--", linewidth=1,
            label=f"No-slip closest approach (t={impact_time:.2f})", zorder=1,
        )
    handles.append(marker)
    for ax, title, ylabel in zip(
        axes,
        (
            "Dissipation in the strip",
            "Share of total dissipation",
            "Energy dissipated in the strip",
        ),
        (
            "Energy loss per unit time",
            "Instantaneous share (%)",
            "% of initial kinetic energy",
        ),
    ):
        ax.set(title=title, ylabel=ylabel, xlabel="Time", xlim=(0, 1.2), ylim=(0, None))
        ax.grid(alpha=0.2)
    fig.suptitle("Vortex–wall collision  |  Within L/16 of the upper wall", fontsize=15)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "vortex_wall_dissipation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
