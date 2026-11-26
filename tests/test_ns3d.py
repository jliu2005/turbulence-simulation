import numpy as np
import pytest

from src.ns3d import NavierStokes3D


def test_fft_roundtrip():
    solver = NavierStokes3D(N=8)
    field = np.random.default_rng(42).normal(size=(3, 8, 8, 8))
    recovered = solver.ifft3(solver.fft3(field))
    np.testing.assert_allclose(recovered, field, atol=1e-12)


def test_incompressibility():
    solver = NavierStokes3D(N=8)
    field = np.random.default_rng(42).normal(size=(3, 8, 8, 8))
    solver.set_initial_velocity(field)

    for _ in range(5):
        u_hat = solver.fft3(solver.velocity())
        div_hat = 1j * np.sum(solver.Kvec * u_hat, axis=0)
        divergence = solver.ifft3(div_hat)
        np.testing.assert_allclose(divergence, 0.0, atol=1e-12)
        solver.step(dt=0.01)


def test_taylor_green_initial_energy():
    solver = NavierStokes3D(N=16)
    solver.set_taylor_green()
    assert solver.kinetic_energy() == pytest.approx(0.125, abs=1e-12)


def test_viscous_decay():
    solver = NavierStokes3D(N=8, nu=0.1)
    initial = np.zeros((3, 8, 8, 8))
    initial[0] = np.sin(solver.Y)
    solver.set_initial_velocity(initial)

    dt = 0.01
    steps = 20
    for _ in range(steps):
        solver.step(dt)

    exact = initial * np.exp(-solver.nu * steps * dt)
    np.testing.assert_allclose(solver.velocity(), exact, rtol=0, atol=1e-7)


def test_energy_decay():
    solver = NavierStokes3D(N=16, nu=0.1)
    solver.set_taylor_green()
    previous = solver.kinetic_energy()

    for _ in range(10):
        solver.step(dt=0.01)
        current = solver.kinetic_energy()
        assert 0 <= current < previous
        previous = current


def test_invalid_input():
    solver = NavierStokes3D(N=8)
    with pytest.raises(ValueError):
        solver.set_initial_velocity(np.zeros((8, 8, 8)))
    with pytest.raises(ValueError):
        solver.step(dt=0)
    with pytest.raises(ValueError):
        NavierStokes3D(N=7)
    with pytest.raises(ValueError):
        NavierStokes3D(nu=-0.1)
