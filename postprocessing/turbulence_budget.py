"""
Post-processing diagnostics for three-dimensional turbulence simulations.

This module turns a saved velocity field ``u(x, y, z, t)`` into the
resolved kinetic-energy budget: kinetic energy, production, transport and
viscous dissipation, together with a global energy-closure diagnostic.

The derivatives are computed spectrally (Fourier), so the analysis assumes a
uniform, triply periodic Cartesian grid -- the same configuration produced by
the Taylor-Green vortex DNS in ``Turb_Dissipation_simulation.ipynb``.

Example
-------
>>> import numpy as np
>>> from postprocessing import plot_turbulence_budget
>>> dns = np.load("taylor_green_dns.npz")            # doctest: +SKIP
>>> results, fig, axes = plot_turbulence_budget(     # doctest: +SKIP
...     velocity=dns["velocity"],
...     nu=float(dns["nu"]),
...     times=dns["times"],
...     domain_lengths=tuple(dns["domain_lengths"]),
...     component_axis=1,
... )

Dependencies
------------
numpy, scipy, matplotlib (see ``requirements.txt``).
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sfft

__all__ = ["plot_turbulence_budget"]


def plot_turbulence_budget(
    velocity,
    nu,
    dt=None,
    times=None,
    domain_lengths=(2 * np.pi, 2 * np.pi, 2 * np.pi),
    component_axis=1,
    dealias=True,
    show=True,
    save_prefix=None,
):
    """
    Calculate and plot the time evolution of the kinetic energy budget
    from a saved three-dimensional velocity field.

    Parameters
    ----------
    velocity : np.ndarray
        Velocity snapshots in physical space.

        Expected shape:
            (nt, 3, nx, ny, nz)

        Alternatively, use component_axis=-1 for:
            (nt, nx, ny, nz, 3)

        Components are ordered [u, v, w]. Integer or float32 input is
        converted to float64 internally, which roughly doubles the memory
        footprint of the array; see the notes in the project README before
        analysing very large snapshot files.

    nu : float
        Kinematic viscosity [m^2/s].

    dt : float, optional
        Time interval between snapshots [s].
        Required if times is not provided.

    times : np.ndarray, optional
        Actual time corresponding to each snapshot [s].
        Use this instead of dt for irregularly spaced snapshots.

    domain_lengths : tuple
        Physical domain dimensions (Lx, Ly, Lz) [m].
        Assumes a uniform, periodic Cartesian grid.

    component_axis : int
        Axis containing the three velocity components.
        Default: 1.

    dealias : bool
        Apply the 2/3-rule spectral dealiasing filter
        to nonlinear products and velocity fields.

    show : bool
        Display plots if True. If False, no figure is displayed, which is
        what you want when calling this from a script or a test.

    save_prefix : str, optional
        If given, every figure is written to disk as
        ``<save_prefix><name>.png`` with names ``budget``, ``closure`` and
        ``terms``. Figures are also returned in ``results["figures"]`` so
        callers can save or restyle them themselves.

    Returns
    -------
    results : dict
        Time histories of:
            KE
            production
            transport_mean
            transport_rms
            dissipation
            cumulative_dissipation
            energy_budget_residual
        plus
            figures
                dict of the matplotlib figures that were created
                ("budget", "closure", "terms").

    fig : matplotlib.figure.Figure
        Figure containing the four-panel turbulence budget plots.

    axes : np.ndarray
        Matplotlib axes of ``fig``.

    Notes
    -----
    Uses Fourier spectral derivatives and assumes periodic
    boundary conditions.

    Production follows the domain-mean decomposition used in
    the original Taylor-Green calculation. It is set to zero.
    This is not a general mean-shear production calculation: the Taylor-Green
    initial condition has zero mean velocity, and a triply periodic domain
    without body forces keeps the domain-averaged momentum constant, so the
    production of resolved kinetic energy by a mean shear is identically zero
    for this configuration. For wall-bounded or channel flows this term must
    be computed from the mean velocity profile instead of being hard-coded.

    Transport includes pressure transport and viscous
    diffusion. Its signed volume average should be
    approximately zero for a periodic domain.

    Dissipation:
        epsilon = 2 * nu * S_ij * S_ij

    Transport:
        T = -div[(K+p)u] + nu * laplacian(K)

    Kinetic energy:
        K = 0.5 * (u^2 + v^2 + w^2)

    The pressure is recovered from a Poisson solve,
    ``laplacian(p) = -d_i(u_j) d_j(u_i)``, with the mean removed by
    construction (the k = 0 mode is set to zero).
    """

    # --------------------------------------------------------
    # 1. INPUT VALIDATION
    # --------------------------------------------------------

    velocity = np.asarray(velocity, dtype=float)

    if velocity.ndim != 5:
        raise ValueError(
            "velocity must be a 5D array: (nt, 3, nx, ny, nz) or (nt, nx, ny, nz, 3)."
        )

    velocity = np.moveaxis(velocity, component_axis, 1)

    nt, ncomp, nx, ny, nz = velocity.shape

    if ncomp != 3:
        raise ValueError("Exactly three velocity components are required.")

    if nt < 1 or min(nx, ny, nz) < 2:
        raise ValueError("Invalid number of snapshots or grid dimensions.")

    if nu < 0:
        raise ValueError("Kinematic viscosity cannot be negative.")

    if times is None:
        if dt is None or dt <= 0:
            raise ValueError("Provide a positive dt or an array of times.")

        times = np.arange(nt) * dt

    else:
        times = np.asarray(times, dtype=float)

        if times.shape != (nt,):
            raise ValueError("times must contain one value per snapshot.")

        if not np.all(np.isfinite(times)):
            raise ValueError("times must contain finite values.")

        if nt > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing.")

    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity contains NaN or infinite values.")

    Lx, Ly, Lz = domain_lengths

    if min(Lx, Ly, Lz) <= 0:
        raise ValueError("Domain lengths must be positive.")

    # --------------------------------------------------------
    # 2. FOURIER GRID
    # --------------------------------------------------------

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=Lz / nz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    K2 = KX**2 + KY**2 + KZ**2

    K2inv = np.zeros_like(K2)

    K2inv[K2 > 0] = 1.0 / K2[K2 > 0]

    # --------------------------------------------------------
    # 3. DEALIASING MASK
    # --------------------------------------------------------

    if dealias:
        ix = np.fft.fftfreq(nx) * nx
        iy = np.fft.fftfreq(ny) * ny
        iz = np.fft.rfftfreq(nz) * nz

        kcut_x = nx // 3
        kcut_y = ny // 3
        kcut_z = nz // 3

        IX, IY, IZ = np.meshgrid(ix, iy, iz, indexing="ij")

        mask = (np.abs(IX) <= kcut_x) & (np.abs(IY) <= kcut_y) & (IZ <= kcut_z)

    else:
        mask = np.ones_like(K2, dtype=bool)

    # --------------------------------------------------------
    # 4. FOURIER TRANSFORM OPERATORS
    # --------------------------------------------------------

    def fft3(f):
        return sfft.rfftn(f, axes=(-3, -2, -1))

    def ifft3(fh):
        return sfft.irfftn(fh, s=(nx, ny, nz), axes=(-3, -2, -1))

    # --------------------------------------------------------
    # 5. STORAGE
    # --------------------------------------------------------

    KE_history = np.zeros(nt)
    production_history = np.zeros(nt)
    transport_mean_history = np.zeros(nt)
    transport_rms_history = np.zeros(nt)
    dissipation_history = np.zeros(nt)

    # --------------------------------------------------------
    # 6. CALCULATE BUDGET AT EVERY SNAPSHOT
    # --------------------------------------------------------

    for n in range(nt):
        # Physical-space velocity
        u = velocity[n]

        # Fourier-space velocity
        uh = fft3(u) * mask

        # Reconstruct filtered physical-space velocity
        u = ifft3(uh)

        # ----------------------------------------------------
        # KINETIC ENERGY
        # ----------------------------------------------------

        K = 0.5 * np.sum(u**2, axis=0)

        KE_history[n] = np.mean(K)

        # ----------------------------------------------------
        # PRODUCTION
        # ----------------------------------------------------

        # Domain-mean decomposition:
        # no imposed mean-flow production.
        production_history[n] = 0.0

        # ----------------------------------------------------
        # VELOCITY GRADIENTS
        # ----------------------------------------------------

        grad = np.empty((3, 3, nx, ny, nz))

        wave_numbers = [KX, KY, KZ]

        for i in range(3):
            for j, Kj in enumerate(wave_numbers):
                grad[i, j] = ifft3(1j * Kj * uh[i])

        # ----------------------------------------------------
        # STRAIN-RATE TENSOR
        # ----------------------------------------------------

        S = 0.5 * (grad + np.swapaxes(grad, 0, 1))

        # ----------------------------------------------------
        # DISSIPATION
        # ----------------------------------------------------

        epsilon_local = 2 * nu * np.sum(S**2, axis=(0, 1))

        dissipation_history[n] = np.mean(epsilon_local)

        # ----------------------------------------------------
        # PRESSURE POISSON EQUATION
        # ----------------------------------------------------

        pressure_source = np.zeros((nx, ny, nz))

        for i in range(3):
            for j in range(3):
                pressure_source -= grad[i, j] * grad[j, i]

        pressure_source_hat = fft3(pressure_source) * mask

        pressure_hat = -pressure_source_hat * K2inv

        pressure = ifft3(pressure_hat)

        # ----------------------------------------------------
        # KINETIC ENERGY TRANSPORT
        # ----------------------------------------------------

        energy_flux = (K + pressure) * u

        flux_hat = fft3(energy_flux) * mask

        divergence_hat = 1j * (KX * flux_hat[0] + KY * flux_hat[1] + KZ * flux_hat[2])

        K_hat = fft3(K) * mask

        laplacian_K = ifft3(-K2 * K_hat)

        transport_local = -ifft3(divergence_hat) + nu * laplacian_K

        transport_mean_history[n] = np.mean(transport_local)

        transport_rms_history[n] = np.sqrt(np.mean(transport_local**2))

    # --------------------------------------------------------
    # 7. CUMULATIVE DISSIPATION
    # --------------------------------------------------------

    cumulative_dissipation = np.zeros(nt)

    if nt > 1:
        cumulative_dissipation[1:] = np.cumsum(
            0.5 * (dissipation_history[1:] + dissipation_history[:-1]) * np.diff(times)
        )

    # --------------------------------------------------------
    # 8. ENERGY BUDGET CLOSURE
    # --------------------------------------------------------

    energy_budget_residual = KE_history + cumulative_dissipation - KE_history[0]

    # --------------------------------------------------------
    # 9. PACKAGE RESULTS
    # --------------------------------------------------------

    results = {
        "times": times,
        "KE": KE_history,
        "production": production_history,
        "transport_mean": transport_mean_history,
        "transport_rms": transport_rms_history,
        "dissipation": dissipation_history,
        "cumulative_dissipation": cumulative_dissipation,
        "energy_budget_residual": energy_budget_residual,
    }

    # --------------------------------------------------------
    # 10. PLOT TURBULENCE BUDGET
    # --------------------------------------------------------

    # A production term that is identically zero would be plotted as a flat
    # line on an auto-scaled axis; use the dissipation scale instead so the
    # panel can be compared visually with the other budget terms.
    production_scale = max(
        float(np.max(np.abs(dissipation_history))),
        float(np.max(np.abs(transport_rms_history))),
        1e-30,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")

    # Production
    ax = axes[0, 0]

    ax.plot(times, production_history, linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylim(-production_scale, production_scale)

    ax.set_title("Turbulent Kinetic Energy Production")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Production [m²/s³]")
    ax.text(
        0.03,
        0.90,
        "Zero by construction:\nno mean shear in this\nperiodic configuration",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="black",
    )
    ax.grid(alpha=0.3)

    # Transport
    ax = axes[0, 1]

    ax.plot(times, transport_rms_history, linewidth=2, label="RMS local transport")

    ax.plot(
        times,
        transport_mean_history,
        "--",
        linewidth=1.5,
        label="Mean signed transport",
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title("Kinetic Energy Transport")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Transport [m²/s³]")
    ax.legend()
    ax.grid(alpha=0.3)

    # Dissipation
    ax = axes[1, 0]

    ax.plot(times, dissipation_history, linewidth=2)

    ax.set_title("Viscous Dissipation")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Mean dissipation $\epsilon$ [m²/s³]")
    ax.grid(alpha=0.3)

    # Global energy budget
    ax = axes[1, 1]

    ax.plot(times, KE_history, linewidth=2, label="Kinetic energy")

    ax.plot(times, cumulative_dissipation, linewidth=2, label="Cumulative dissipation")

    ax.plot(
        times,
        KE_history + cumulative_dissipation,
        "--",
        linewidth=1.5,
        label="KE + cumulative dissipation",
    )

    ax.axhline(KE_history[0], color="black", linestyle=":", label="Initial KE")

    ax.set_title("Global Energy Budget")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [m²/s²]")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Turbulence Budget | Grid: {nx} × {ny} × {nz}", fontsize=15)

    # Energy closure diagnostic
    fig_residual, ax_residual = plt.subplots(figsize=(9, 4))

    ax_residual.plot(times, energy_budget_residual, linewidth=2)

    ax_residual.axhline(0, color="black", linestyle="--")

    ax_residual.set_title("Energy Budget Closure Error")
    ax_residual.set_xlabel("Time [s]")
    ax_residual.set_ylabel("Energy residual [m²/s²]")
    ax_residual.grid(alpha=0.3)

    fig_residual.tight_layout()

    # ============================================================
    # ADDITIONAL FIGURE: PRODUCTION, TRANSPORT, AND DISSIPATION
    # ============================================================

    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))

    ax_combined.plot(times, production_history, label="Production", linewidth=2)

    ax_combined.plot(
        times, transport_rms_history, label="RMS local transport", linewidth=2
    )

    ax_combined.plot(times, dissipation_history, label="Dissipation", linewidth=2)

    ax_combined.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax_combined.set_xlabel("Time [s]")
    ax_combined.set_ylabel("Budget term magnitude [m²/s³]")
    ax_combined.set_title("Production, RMS Transport, and Dissipation")

    ax_combined.legend()
    ax_combined.grid(alpha=0.3)

    fig_combined.tight_layout()

    # --------------------------------------------------------
    # 11. SAVE / DISPLAY
    # --------------------------------------------------------

    figures = {"budget": fig, "closure": fig_residual, "terms": fig_combined}

    results["figures"] = figures

    if save_prefix is not None:
        for name, figure in figures.items():
            figure.savefig(f"{save_prefix}{name}.png", dpi=150)

    if show:
        plt.show()

    return results, fig, axes
