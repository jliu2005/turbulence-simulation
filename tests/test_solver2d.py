import numpy as np
import pytest

from src.ns2d import NavierStokes2D
from src.ns2d_walls import NavierStokes2DWalls
from src.solver2d import Wall, make_solver_2d


@pytest.mark.parametrize("case", ["periodic", "free_slip", "no_slip", "moving_lid"])
def test_constructor_matches_direct_run(case):
    params = dict(N=24 if case == "periodic" else 25, L=2.0, nu=0.012)
    if case == "periodic":
        boundaries = "periodic"
        direct = NavierStokes2D(**params)
    else:
        boundaries = dict.fromkeys(
            ("left", "right", "bottom", "top"), Wall("free_slip")
        )
        if case != "free_slip":
            boundaries["top"] = Wall(
                "no_slip", speed=0.4 if case == "moving_lid" else 0
            )
        direct = NavierStokes2DWalls(**params, walls=boundaries)
    solver = make_solver_2d(**params, boundaries=boundaries)
    assert type(solver) is type(direct)
    np.testing.assert_array_equal(solver.X, direct.X)
    np.testing.assert_array_equal(solver.Y, direct.Y)
    omega = np.sin(np.pi * direct.X) * np.sin(np.pi * direct.Y)
    for run in (solver, direct):
        run.set_initial_vorticity(omega)
        run.step(0.001)
    np.testing.assert_array_equal(solver.vorticity(), direct.vorticity())
    np.testing.assert_array_equal(solver.velocity(), direct.velocity())
    assert solver.kinetic_energy() == direct.kinetic_energy()


@pytest.mark.parametrize("boundaries", ["channel", {"top": Wall()}])
def test_constructor_rejects_unsupported_boundaries(boundaries):
    with pytest.raises(ValueError):
        make_solver_2d(N=16, L=1, nu=0.01, boundaries=boundaries)
