"""Construct a 2D Navier–Stokes solver for a periodic or walled square."""

from collections.abc import Mapping
from typing import Literal

from src.ns2d import NavierStokes2D
from src.ns2d_walls import NavierStokes2DWalls, Wall


def make_solver_2d(
    *, N: int, L: float, nu: float, boundaries: Literal["periodic"] | Mapping[str, Wall]
) -> NavierStokes2D | NavierStokes2DWalls:
    """Build an uninitialized solver; set its initial vorticity on its own grid.

    For a periodic square, pass "periodic" (grid spacing L/N). For an
    impermeable square, specify a Wall on every side (spacing L/(N-1)).
    """
    if isinstance(boundaries, str):
        if boundaries == "periodic":
            return NavierStokes2D(N=N, L=L, nu=nu)
        raise ValueError("The only supported string boundary setting is 'periodic'")
    if not isinstance(boundaries, Mapping):
        raise TypeError("boundaries must be 'periodic' or a mapping of sides to walls")
    if set(boundaries) != {"left", "right", "bottom", "top"}:
        raise ValueError("Specify exactly four wall sides: left, right, bottom, top")
    return NavierStokes2DWalls(N=N, L=L, nu=nu, walls=boundaries)
