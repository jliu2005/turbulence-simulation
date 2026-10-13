import numpy as np
from src.ns2d import NavierStokes2D


def initial_taylor_green(s):
    # u=sin(x)cos(y), v=-cos(x)sin(y), omega=2sin(x)sin(y)
    return 2 * np.sin(s.X) * np.sin(s.Y)


def test_velocity_recovery_and_incompressibility():
    s = NavierStokes2D(N=32)
    s.set_initial_vorticity(initial_taylor_green(s))
    u, v = s.velocity()
    np.testing.assert_allclose(u, np.sin(s.X) * np.cos(s.Y), atol=1e-12)
    np.testing.assert_allclose(v, -np.cos(s.X) * np.sin(s.Y), atol=1e-12)
    div = np.fft.ifft2(1j * s.KX * np.fft.fft2(u)
                         + 1j * s.KY * np.fft.fft2(v)).real
    assert np.max(np.abs(div)) < 1e-12


def test_taylor_green_exact_decay():
    nu, t, dt = 0.05, 0.2, 0.002
    s = NavierStokes2D(N=32, nu=nu)
    s.set_initial_vorticity(initial_taylor_green(s))
    for _ in range(round(t / dt)):
        s.step(dt)
    expected = initial_taylor_green(s) * np.exp(-2 * nu * t)
    np.testing.assert_allclose(s.vorticity(), expected, atol=1e-9)


def test_energy_decreases_without_forcing():
    s = NavierStokes2D(N=32, nu=0.05)
    s.set_initial_vorticity(initial_taylor_green(s))
    e0 = s.kinetic_energy()
    for _ in range(50):
        s.step(0.002)
    assert 0 < s.kinetic_energy() < e0


def test_initial_shape_rejected():
    s = NavierStokes2D(N=32)
    try:
        s.set_initial_vorticity(np.zeros((16, 16)))
    except ValueError:
        pass
    else:
        raise AssertionError('Expected ValueError')
