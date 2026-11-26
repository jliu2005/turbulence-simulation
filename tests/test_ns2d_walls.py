"""Verification cases for the node-centered, wall-bounded 2D solver."""

import numpy as np
import pytest

from src.ns2d_walls import NavierStokes2DWalls, Wall

SIDES = ("left", "right", "bottom", "top")
FREE_SLIP = {side: Wall("free_slip") for side in SIDES}


def test_poisson_inverts_five_point_laplacian_with_zero_boundary():
    s = NavierStokes2DWalls(N=24, L=2.3, walls=FREE_SLIP)
    x, y = s.X / s.L, s.Y / s.L
    exact = x * (1 - x) * y * (1 - y) * (1 + x * y)
    h = s.L / (s.N - 1)
    omega = np.zeros_like(x)
    omega[1:-1, 1:-1] = (
        4 * exact[1:-1, 1:-1]
        - exact[2:, 1:-1]
        - exact[:-2, 1:-1]
        - exact[1:-1, 2:]
        - exact[1:-1, :-2]
    ) / h**2
    s.set_initial_vorticity(omega)
    np.testing.assert_allclose(s.streamfunction(), exact, atol=2e-14)


def test_free_slip_mode_converges_to_exact_decay_and_velocity():
    errors = []
    for n in (17, 33, 65):
        s = NavierStokes2DWalls(N=n, L=2.0, nu=0.1, walls=FREE_SLIP)
        k = np.pi / s.L
        mode = np.sin(k * s.X) * np.sin(k * s.Y)
        omega0 = 2 * k**2 * mode
        s.set_initial_vorticity(omega0)
        for _ in range(40):
            s.step(0.001)
        amplitude = np.exp(-2 * s.nu * k**2 * 0.04)
        expected_u = amplitude * k * np.sin(k * s.X) * np.cos(k * s.Y)
        expected_v = -amplitude * k * np.cos(k * s.X) * np.sin(k * s.Y)
        u, v = s.velocity()
        inside = np.s_[1:-1, 1:-1]
        errors.append(
            [
                np.max(np.abs(s.vorticity() - amplitude * omega0)),
                np.max(np.abs(u[inside] - expected_u[inside])),
                np.max(np.abs(v[inside] - expected_v[inside])),
            ]
        )
        # Nonzero tangential motion is permitted even on a stationary free-slip wall.
        assert u[n // 2, 0] == pytest.approx(expected_u[n // 2, 0], rel=0.03)
        assert np.max(np.abs(u[[0, -1], :])) == 0
        assert np.max(np.abs(v[:, [0, -1]])) == 0
    errors = np.array(errors)
    assert np.all(errors[:-1] / errors[1:] > 3.6)
    assert s.kinetic_energy() == pytest.approx(amplitude**2 * k**2 / 4, rel=0.003)
    assert s.dissipation() == pytest.approx(s.nu * amplitude**2 * k**4, rel=0.005)


def test_nonlinear_advection_of_two_modes_and_discrete_divergence():
    s = NavierStokes2DWalls(N=65, L=np.pi, nu=0, walls=FREE_SLIP)
    x, y = s.X, s.Y
    a = 0.3
    # psi = (sin(x) + a sin(2x)) sin(y). The two modes advect one another.
    omega = (2 * np.sin(x) + 5 * a * np.sin(2 * x)) * np.sin(y)
    u = (np.sin(x) + a * np.sin(2 * x)) * np.cos(y)
    v = -(np.cos(x) + 2 * a * np.cos(2 * x)) * np.sin(y)
    wx = (2 * np.cos(x) + 10 * a * np.cos(2 * x)) * np.sin(y)
    wy = (2 * np.sin(x) + 5 * a * np.sin(2 * x)) * np.cos(y)
    expected = -u * wx - v * wy
    s.set_initial_vorticity(omega)
    dt = 1e-5
    s.step(dt)
    approx = (s.vorticity() - omega) / dt
    assert np.linalg.norm(approx - expected) < 0.01 * np.linalg.norm(expected)
    u, v = s.velocity()
    h = s.L / (s.N - 1)
    div = (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2]) / (2 * h)
    np.testing.assert_allclose(div, 0, atol=1e-12)
    assert s.dissipation() == 0


@pytest.mark.parametrize(
    "side, axis, endpoint, vorticity_sign",
    [("left", 0, 0, -1), ("right", 0, -1, 1), ("bottom", 1, 0, 1), ("top", 1, -1, -1)],
)
def test_signed_wall_speed_and_vorticity_on_each_side(
    side, axis, endpoint, vorticity_sign
):
    speed = -0.7
    s = NavierStokes2DWalls(
        N=17, L=2, walls={**FREE_SLIP, side: Wall("no_slip", speed)}
    )
    u, v = s.velocity()
    tangent, normal = (v, u) if axis == 0 else (u, v)
    np.testing.assert_allclose(tangent.take(endpoint, axis=axis)[1:-1], speed)
    np.testing.assert_allclose(normal.take(endpoint, axis=axis)[1:-1], 0)
    h = s.L / (s.N - 1)
    expected = vorticity_sign * 2 * speed / h
    np.testing.assert_allclose(s.vorticity().take(endpoint, axis=axis)[1:-1], expected)
    s.step(0.001)
    np.testing.assert_allclose(s.vorticity().take(-1 - endpoint, axis=axis)[1:-1], 0)


def test_cavity_recirculation_and_fourth_order_time_refinement():
    states = []
    for steps in (8, 16, 32):
        s = NavierStokes2DWalls(N=17, nu=0.01, walls={"top": Wall("no_slip", 1)})
        for _ in range(steps):
            s.step(0.16 / steps)
        u, v = s.velocity()
        # Rightward top lid drives a clockwise recirculation in the interior.
        assert u[8, -2] > 0 and u[8, 5] < 0
        assert v[4, 10] > 0 and v[-5, 10] < 0
        states.append(s.vorticity()[1:-1, 1:-1])
    coarse = np.linalg.norm(states[0] - states[1])
    fine = np.linalg.norm(states[1] - states[2])
    assert coarse / fine > 13


def test_stationary_no_slip_energy_budget_refines():
    residuals = []
    for n in (17, 33, 65):
        s = NavierStokes2DWalls(N=n, nu=0.01)
        x, y = np.pi * s.X, np.pi * s.Y
        # Smooth streamfunction 0.02 sin²(pi x) sin²(pi y) has zero wall velocity.
        omega0 = (
            -0.04
            * np.pi**2
            * (np.cos(2 * x) * np.sin(y) ** 2 + np.cos(2 * y) * np.sin(x) ** 2)
        )
        s.set_initial_vorticity(omega0)
        initial_ke = s.kinetic_energy()
        dt = 0.001
        dissipation = [s.dissipation()]
        for _ in range(40):
            s.step(dt)
            dissipation.append(s.dissipation())
        energy_loss = initial_ke - s.kinetic_energy()
        assert energy_loss > 0
        losses = np.array(dissipation)
        integral = 0.5 * dt * (losses[:-1] + losses[1:]).sum()
        residuals.append(abs(energy_loss - integral) / energy_loss)
    assert residuals[0] > residuals[1] > residuals[2]
    assert residuals[-1] < 0.01


@pytest.mark.parametrize(
    "kwargs",
    [{"N": 2}, {"N": 8.5}, {"L": 0}, {"L": np.inf}, {"nu": -0.1}, {"nu": np.nan}],
)
def test_invalid_grid_and_viscosity(kwargs):
    with pytest.raises((TypeError, ValueError)):
        NavierStokes2DWalls(**kwargs)


@pytest.mark.parametrize(
    "side, kind, speed",
    [
        ("front", "no_slip", 0),
        ("left", "unknown", 0),
        ("top", "free_slip", 1),
        ("bottom", "no_slip", np.inf),
    ],
)
def test_invalid_wall_configuration(side, kind, speed):
    with pytest.raises((TypeError, ValueError)):
        NavierStokes2DWalls(N=9, walls={side: Wall(kind, speed)})


@pytest.mark.parametrize("dt", [0, -0.001, np.nan, np.inf])
def test_invalid_step_preserves_state(dt):
    s = NavierStokes2DWalls(N=9, walls={"left": Wall("no_slip", -1)})
    s.step(0.001)
    before = s.vorticity().copy()
    with pytest.raises((TypeError, ValueError)):
        s.step(dt)
    np.testing.assert_array_equal(s.vorticity(), before)


def test_initial_vorticity_validation_and_wall_reconstruction():
    s = NavierStokes2DWalls(N=9, walls=FREE_SLIP)
    omega = np.ones((9, 9))
    s.set_initial_vorticity(omega)
    expected = np.zeros_like(omega)
    expected[1:-1, 1:-1] = 1
    before = s.vorticity().copy()
    np.testing.assert_allclose(before, expected)
    for field in (omega[:, :-1], omega * np.nan, omega + 1j):
        with pytest.raises((TypeError, ValueError)):
            s.set_initial_vorticity(field)
        np.testing.assert_array_equal(s.vorticity(), before)


def test_steady_cavity_centerlines_against_ghia():
    # Use a small full run in CI so the benchmark data and profile orientations
    # are exercised together with the flow solver.
    from simulations.run_2d_cavity import run

    result = run(n=33, steps_per_sample=32)
    u_error, v_error = result.max_profile_errors()
    # Absolute errors are in units of the lid speed. Both are <0.009 on this grid.
    assert u_error < 0.012
    assert v_error < 0.012
