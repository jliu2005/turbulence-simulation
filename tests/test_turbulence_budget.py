"""
Pytest suite for postprocessing.plot_turbulence_budget.

Run with:

    python -m pytest -q

The tests deliberately use a tiny grid (N = 16) so the whole suite finishes in
a few seconds. They check three kinds of behaviour:

1. Input validation - the function must fail loudly on malformed fields.
2. Exact identities that hold for *any* periodic input, e.g. the domain average
   of a spectral divergence is zero, and zero viscosity means zero dissipation.
3. Physics of a real (short) Taylor-Green DNS run: energy decays monotonically,
   dissipation is positive, and the global budget closes to solver tolerance.
"""

import matplotlib

matplotlib.use("Agg")  # headless: never open windows during tests

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from postprocessing import plot_turbulence_budget  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def taylor_green_ic(N, domain_length=2 * np.pi):
    """Analytic Taylor-Green vortex, shape (3, N, N, N). Divergence-free."""
    x = np.arange(N) * domain_length / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    return np.array(
        [
            np.sin(X) * np.cos(Y) * np.cos(Z),
            -np.cos(X) * np.sin(Y) * np.cos(Z),
            np.zeros_like(X),
        ]
    )


def run_mini_dns(N=16, nu=1 / 50, t_end=5.0, dt=0.05, snap_every=5):
    """
    Short pseudo-spectral Taylor-Green run.

    Returns
    -------
    velocity : (nt, 3, N, N, N) float64
    times : (nt,)
    """
    import scipy.fft as sfft

    L = 2 * np.pi
    x = np.arange(N) * L / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    Kvec = np.array([KX, KY, KZ])
    K2 = KX**2 + KY**2 + KZ**2
    K2inv = np.zeros_like(K2)
    K2inv[K2 > 0] = 1.0 / K2[K2 > 0]

    kcut = N // 3
    mask = (np.abs(KX) <= kcut) & (np.abs(KY) <= kcut) & (np.abs(KZ) <= kcut)

    def fft3(f):
        return sfft.rfftn(f, axes=(-3, -2, -1))

    def ifft3(fh):
        return sfft.irfftn(fh, s=(N, N, N), axes=(-3, -2, -1))

    def curl_hat(uh):
        return 1j * np.array(
            [
                KY * uh[2] - KZ * uh[1],
                KZ * uh[0] - KX * uh[2],
                KX * uh[1] - KY * uh[0],
            ]
        )

    def rhs(uh):
        u = ifft3(uh)
        w = ifft3(curl_hat(uh))
        cross = np.array(
            [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ]
        )
        nh = fft3(cross) * mask
        nh -= Kvec * (np.sum(Kvec * nh, axis=0) * K2inv)
        return nh - nu * K2 * uh

    uh = fft3(taylor_green_ic(N)) * mask

    nsteps = int(round(t_end / dt))
    nsnap = nsteps // snap_every + 1

    velocity = np.empty((nsnap, 3, N, N, N))
    times = np.empty(nsnap)

    idx = 0
    for n in range(nsteps + 1):
        if n % snap_every == 0:
            velocity[idx] = ifft3(uh)
            times[idx] = n * dt
            idx += 1
        if n < nsteps:
            k1_ = rhs(uh)
            k2_ = rhs(uh + dt * k1_)
            uh = uh + 0.5 * dt * (k1_ + k2_)

    return velocity[:idx], times[:idx]


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Each call to plot_turbulence_budget creates three figures that stay open
    until the caller closes them (intentional: they are returned so they can be
    saved or restyled). Close them between tests so matplotlib's figure
    registry does not grow across the suite.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="module")
def frozen_field():
    """The same Taylor-Green snapshot repeated 5 times (no evolution)."""
    u0 = taylor_green_ic(16)
    velocity = np.stack([u0] * 5)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return velocity, times


# ---------------------------------------------------------------------------
# 1. API contract
# ---------------------------------------------------------------------------


def test_returns_documented_keys(frozen_field):
    velocity, times = frozen_field

    results, fig, axes = plot_turbulence_budget(
        velocity=velocity, nu=1e-3, times=times, show=False
    )

    expected = {
        "times",
        "KE",
        "production",
        "transport_mean",
        "transport_rms",
        "dissipation",
        "cumulative_dissipation",
        "energy_budget_residual",
        "figures",
    }
    assert set(results) == expected

    for key in expected - {"times", "figures"}:
        assert results[key].shape == (5,), key

    assert set(results["figures"]) == {"budget", "closure", "terms"}
    assert fig is results["figures"]["budget"]
    assert axes.shape == (2, 2)


def test_arrays_are_finite(frozen_field):
    velocity, times = frozen_field

    results, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1e-3, times=times, show=False
    )

    for key, value in results.items():
        if key == "figures":
            continue
        assert np.all(np.isfinite(value)), key


# ---------------------------------------------------------------------------
# 2. Identities that must hold for any periodic input
# ---------------------------------------------------------------------------


def test_viscosity_zero_gives_zero_dissipation_and_closure(frozen_field):
    velocity, times = frozen_field

    results, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=0.0, times=times, show=False
    )

    np.testing.assert_allclose(results["dissipation"], 0.0, atol=1e-15)
    np.testing.assert_allclose(results["cumulative_dissipation"], 0.0, atol=1e-15)
    np.testing.assert_allclose(results["energy_budget_residual"], 0.0, atol=1e-15)


def test_mean_transport_vanishes_on_periodic_domain(frozen_field):
    """The domain average of a spectral divergence is exactly the k=0 mode."""
    velocity, times = frozen_field

    results, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1e-3, times=times, show=False
    )

    np.testing.assert_allclose(results["transport_mean"], 0.0, atol=1e-14)


def test_frozen_field_closure_reduces_to_cumulative_dissipation(frozen_field):
    """If the field never changes, KE is constant, so residual == int(eps dt)."""
    velocity, times = frozen_field

    results, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1e-3, times=times, show=False
    )

    np.testing.assert_allclose(results["KE"], results["KE"][0], rtol=1e-12)
    np.testing.assert_allclose(
        results["energy_budget_residual"],
        results["cumulative_dissipation"],
        rtol=1e-12,
        atol=1e-15,
    )
    # Constant field => constant dissipation => trapezoid is exact.
    eps = results["dissipation"][0]
    np.testing.assert_allclose(
        results["cumulative_dissipation"], eps * times, rtol=1e-12, atol=1e-15
    )


def test_production_is_identically_zero(frozen_field):
    """Documented behaviour: no mean shear in this configuration."""
    velocity, times = frozen_field

    results, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1e-3, times=times, show=False
    )

    np.testing.assert_array_equal(results["production"], np.zeros(5))


def test_initial_kinetic_energy_matches_theory():
    """The Taylor-Green initial condition has exactly KE = 0.125."""
    u0 = taylor_green_ic(32)

    results, _, _ = plot_turbulence_budget(
        velocity=u0[np.newaxis, ...], nu=0.0, dt=1.0, show=False
    )

    np.testing.assert_allclose(results["KE"][0], 0.125, rtol=1e-12)


def test_dealiasing_removes_energy_near_the_cutoff():
    """A field with content above 2/3 of the cutoff must lose energy."""
    rng = np.random.default_rng(0)
    velocity = rng.normal(size=(1, 3, 24, 24, 24))

    kwargs = dict(nu=1e-3, dt=1.0, show=False)
    full, _, _ = plot_turbulence_budget(velocity=velocity, dealias=False, **kwargs)
    cut, _, _ = plot_turbulence_budget(velocity=velocity, dealias=True, **kwargs)

    assert cut["KE"][0] < full["KE"][0]


# ---------------------------------------------------------------------------
# 3. Physics of a short DNS run
# ---------------------------------------------------------------------------


def test_short_dns_run_closes_the_energy_budget():
    velocity, times = run_mini_dns()

    results, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1 / 50, times=times, show=False
    )

    ke0 = results["KE"][0]

    # Energy must decay, and dissipation must be positive throughout.
    assert results["KE"][-1] < 0.5 * ke0
    assert np.all(results["dissipation"] > 0)

    # dKE/dt = -eps to solver accuracy: the residual stays a small fraction of
    # the initial energy.
    closure_error = np.abs(results["energy_budget_residual"]).max()
    assert closure_error / ke0 < 1e-3, closure_error / ke0

    # Dissipation should peak once turbulence is fully developed and then decay.
    peak = int(np.argmax(results["dissipation"]))
    assert 0 < peak < len(times) - 1


def test_component_axis_last_gives_same_answer():
    velocity, times = run_mini_dns(N=16, t_end=2.0)

    axis_first, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1 / 50, times=times, component_axis=1, show=False
    )

    velocity_last = np.moveaxis(velocity, 1, -1)
    axis_last, _, _ = plot_turbulence_budget(
        velocity=velocity_last, nu=1 / 50, times=times, component_axis=-1, show=False
    )

    for key in ("KE", "dissipation", "transport_rms", "cumulative_dissipation"):
        np.testing.assert_allclose(
            axis_first[key], axis_last[key], rtol=1e-12, atol=1e-15, err_msg=key
        )


def test_dt_and_explicit_times_agree():
    velocity, times = run_mini_dns(N=16, t_end=2.0)

    from_times, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1 / 50, times=times, show=False
    )
    from_dt, _, _ = plot_turbulence_budget(
        velocity=velocity, nu=1 / 50, dt=times[1] - times[0], show=False
    )

    np.testing.assert_allclose(
        from_times["cumulative_dissipation"],
        from_dt["cumulative_dissipation"],
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------


def test_rejects_wrong_number_of_dimensions():
    with pytest.raises(ValueError, match="5D array"):
        plot_turbulence_budget(velocity=np.zeros((4, 8, 8, 8)), nu=1e-3, dt=0.1)


def test_rejects_wrong_component_count():
    with pytest.raises(ValueError, match="three velocity components"):
        plot_turbulence_budget(velocity=np.zeros((2, 2, 8, 8, 8)), nu=1e-3, dt=0.1)


def test_rejects_negative_viscosity():
    u0 = taylor_green_ic(8)[np.newaxis, ...]
    with pytest.raises(ValueError, match="negative"):
        plot_turbulence_budget(velocity=u0, nu=-1e-3, dt=0.1)


def test_rejects_non_finite_velocity():
    velocity, _ = run_mini_dns(N=8, t_end=0.1, snap_every=1)
    velocity[0, 0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        plot_turbulence_budget(velocity=velocity, nu=1e-3, dt=0.1)


def test_requires_dt_or_times():
    u0 = taylor_green_ic(8)[np.newaxis, ...]
    with pytest.raises(ValueError, match="positive dt"):
        plot_turbulence_budget(velocity=u0, nu=1e-3)


def test_rejects_mismatched_times_length():
    velocity, times = run_mini_dns(N=8, t_end=0.2, snap_every=1)
    with pytest.raises(ValueError, match="one value per snapshot"):
        plot_turbulence_budget(velocity=velocity, nu=1e-3, times=times[:-1])


def test_rejects_unsorted_times():
    velocity, times = run_mini_dns(N=8, t_end=0.2, snap_every=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        plot_turbulence_budget(velocity=velocity, nu=1e-3, times=times[::-1].copy())


def test_rejects_non_positive_domain_length():
    u0 = taylor_green_ic(8)[np.newaxis, ...]
    with pytest.raises(ValueError, match="Domain lengths"):
        plot_turbulence_budget(
            velocity=u0, nu=1e-3, dt=0.1, domain_lengths=(0.0, 2 * np.pi, 2 * np.pi)
        )


# ---------------------------------------------------------------------------
# 5. Figure handling
# ---------------------------------------------------------------------------


def test_show_false_creates_figures_without_displaying(frozen_field, monkeypatch):
    """show=False must be respected (it previously was not: plt.show() ran twice)."""
    import matplotlib.pyplot as plt

    called = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(True), raising=True)

    velocity, times = frozen_field
    plot_turbulence_budget(velocity=velocity, nu=1e-3, times=times, show=False)

    assert called == []


def test_save_prefix_writes_all_figures(frozen_field, tmp_path):
    velocity, times = frozen_field

    plot_turbulence_budget(
        velocity=velocity,
        nu=1e-3,
        times=times,
        show=False,
        save_prefix=str(tmp_path / "fig_"),
    )

    for name in ("budget", "closure", "terms"):
        path = tmp_path / f"fig_{name}.png"
        assert path.exists(), name
        assert path.stat().st_size > 5_000, f"{name} looks empty"


def test_single_snapshot_does_not_crash():
    u0 = taylor_green_ic(16)[np.newaxis, ...]

    results, _, _ = plot_turbulence_budget(velocity=u0, nu=1e-3, dt=0.5, show=False)

    assert results["KE"].shape == (1,)
    np.testing.assert_array_equal(results["cumulative_dissipation"], [0.0])
    np.testing.assert_allclose(results["energy_budget_residual"], [0.0], atol=1e-15)


def test_no_figures_left_open():
    """The module must not leak open figures into the caller's session."""
    import matplotlib.pyplot as plt

    u0 = taylor_green_ic(8)[np.newaxis, ...]

    plt.close("all")
    plot_turbulence_budget(velocity=u0, nu=1e-3, dt=1.0, show=False)

    # Three figures are created and intentionally kept alive via results["figures"].
    assert len(plt.get_fignums()) == 3

    plt.close("all")
    assert len(plt.get_fignums()) == 0
