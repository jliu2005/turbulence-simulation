"""Incompressible 2D flow in a square with impermeable straight walls."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
from scipy.fft import dstn, idstn


def _finite_scalar(value):
    return isinstance(value, Real) and np.isfinite(value)


@dataclass(frozen=True)
class Wall:
    """Tangential condition on one side of the square.

    A no-slip wall imposes a constant tangential velocity, positive along x
    on the horizontal sides and along y on the vertical sides. A free-slip
    wall imposes zero tangential stress.
    """

    kind: str = "no_slip"
    speed: float = 0.0

    def __post_init__(self):
        if self.kind not in ("free_slip", "no_slip"):
            raise ValueError("Wall kind must be 'free_slip' or 'no_slip'")
        if not _finite_scalar(self.speed):
            raise ValueError("Wall speed must be a finite real scalar")
        if self.kind == "free_slip" and self.speed != 0:
            raise ValueError("A free-slip wall does not prescribe a speed")


class NavierStokes2DWalls:
    """Unforced vorticity-streamfunction solver on an endpoint-inclusive grid.

    Parameters
    ----------
    N : int
        Number of nodes in each direction, including both boundary nodes.
    L : float
        Side length. Arrays have shape (N, N), indexed by x first, then y.
    nu : float
        Kinematic viscosity. Zero is supported for all-free-slip walls.
    walls : mapping, optional
        Map 'left', 'right', 'bottom', and 'top' to :class:`Wall` objects.
        Omitted sides are stationary no-slip walls.

    The streamfunction satisfies -Laplacian(psi) = omega with zero Dirichlet
    data. Both normal velocities vanish at corners, where either moving
    side's prescribed speed may be discontinuous. Corner vorticity is
    reported as zero and does not enter the interior stencil.
    """

    def __init__(self, N=65, L=1.0, nu=0.01, walls=None):
        if not isinstance(N, (int, np.integer)) or N < 5:
            raise ValueError("N must be an integer >= 5")
        if not _finite_scalar(L) or L <= 0 or not _finite_scalar(nu) or nu < 0:
            raise ValueError("Require finite L > 0 and nu >= 0")

        sides = dict.fromkeys(("left", "right", "bottom", "top"), Wall())
        if walls is not None:
            if not isinstance(walls, Mapping):
                raise TypeError("walls must map side names to Wall objects")
            for side, wall in walls.items():
                if side not in sides:
                    raise ValueError(f"Unknown wall side: {side}")
                if not isinstance(wall, Wall):
                    raise TypeError(f"Expected a Wall for {side}")
                sides[side] = wall
        if nu == 0 and any(wall.kind == "no_slip" for wall in sides.values()):
            raise ValueError(
                "Zero viscosity requires free-slip conditions on all walls"
            )
        self.walls = MappingProxyType(sides)

        self.N, self.L, self.nu = N, L, nu
        self.h = L / (N - 1)
        x = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")

        # Eigenvalues of the nodal five-point negative Laplacian with psi=0.
        modes = np.arange(1, N - 1)
        eigenvalues = (2 / self.h * np.sin(np.pi * modes / (2 * (N - 1)))) ** 2
        self._inverse_laplacian = 1 / (eigenvalues[:, None] + eigenvalues[None, :])
        self._interior = np.zeros((N - 2, N - 2))
        weights = np.ones(N)
        weights[[0, -1]] = 0.5
        self._area_weights = np.outer(weights, weights) / (N - 1) ** 2

    def set_initial_vorticity(self, omega):
        """Initialize the interior from an (N,N) field. Walls supply edge data."""
        field = np.asarray(omega)
        if field.shape != (self.N, self.N) or not np.isrealobj(field):
            raise ValueError("Initial vorticity must be a real array with shape (N, N)")
        field = np.asarray(field, dtype=float)
        if not np.isfinite(field).all():
            raise ValueError("Initial vorticity must be finite")
        self._interior = field[1:-1, 1:-1].copy()

    def _solve_streamfunction(self, interior):
        transformed = dstn(interior, type=1, norm="ortho")
        psi = np.zeros((self.N, self.N))
        psi[1:-1, 1:-1] = idstn(
            transformed * self._inverse_laplacian, type=1, norm="ortho"
        )
        return psi

    def streamfunction(self):
        """Return psi on the full grid, with velocity (psi_y, -psi_x)."""
        return self._solve_streamfunction(self._interior)

    def _wall_vorticity(self, interior, psi):
        omega = np.zeros_like(psi)
        omega[1:-1, 1:-1] = interior

        # On a flat free-slip wall omega=0. For no slip, Thom's formula is
        # -2*(psi_adjacent - h*dpsi/dn_inward)/h**2. Edge endpoints are omitted.
        h = self.h
        left = self.walls["left"]
        if left.kind == "no_slip":
            omega[0, 1:-1] = -2 * (psi[1, 1:-1] + h * left.speed) / h**2
        right = self.walls["right"]
        if right.kind == "no_slip":
            omega[-1, 1:-1] = -2 * (psi[-2, 1:-1] - h * right.speed) / h**2
        bottom = self.walls["bottom"]
        if bottom.kind == "no_slip":
            omega[1:-1, 0] = -2 * (psi[1:-1, 1] - h * bottom.speed) / h**2
        top = self.walls["top"]
        if top.kind == "no_slip":
            omega[1:-1, -1] = -2 * (psi[1:-1, -2] + h * top.speed) / h**2
        return omega

    def vorticity(self):
        """Return interior and reconstructed wall vorticity."""
        return self._wall_vorticity(self._interior, self.streamfunction())

    def velocity(self):
        """Return (u,v) on all nodes, including tangential wall velocities."""
        psi = self.streamfunction()
        psi_x, u = np.gradient(psi, self.h, edge_order=2)
        v = -psi_x
        for side, index in (("left", 0), ("right", -1)):
            u[index, :] = 0
            if self.walls[side].kind == "no_slip":
                v[index, 1:-1] = self.walls[side].speed
        for side, index in (("bottom", 0), ("top", -1)):
            v[:, index] = 0
            if self.walls[side].kind == "no_slip":
                u[1:-1, index] = self.walls[side].speed
        return u, v

    def _rhs(self, interior):
        psi = self._solve_streamfunction(interior)
        omega = self._wall_vorticity(interior, psi)
        h = self.h
        u = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        v = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        omega_x = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2 * h)
        omega_y = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2 * h)
        laplacian = (
            omega[2:, 1:-1]
            + omega[:-2, 1:-1]
            + omega[1:-1, 2:]
            + omega[1:-1, :-2]
            - 4 * interior
        ) / h**2
        return -u * omega_x - v * omega_y + self.nu * laplacian

    def step(self, dt):
        """Take one RK4 step. Choose dt for advective and viscous stability."""
        if not _finite_scalar(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        w = self._interior
        k1 = self._rhs(w)
        k2 = self._rhs(w + 0.5 * dt * k1)
        k3 = self._rhs(w + 0.5 * dt * k2)
        k4 = self._rhs(w + dt * k3)
        updated = w + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(updated).all():
            raise FloatingPointError("Nonfinite RK4 update; check dt and resolution")
        self._interior = updated

    def kinetic_energy(self):
        """Area-averaged kinetic energy per unit mass, by trapezoidal quadrature."""
        u, v = self.velocity()
        return float(0.5 * np.sum(self._area_weights * (u**2 + v**2)))

    def dissipation_density(self):
        """Local viscous dissipation per unit mass, 2 nu S:S, on the full grid."""
        u, v = self.velocity()
        ux, uy = np.gradient(u, self.h, edge_order=2)
        vx, vy = np.gradient(v, self.h, edge_order=2)
        strain_squared = ux**2 + vy**2 + 0.5 * (uy + vx) ** 2
        return 2 * self.nu * strain_squared

    def dissipation(self):
        """Area-averaged viscous dissipation per unit mass."""
        return float(np.sum(self._area_weights * self.dissipation_density()))
