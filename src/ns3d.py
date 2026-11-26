import numpy as np
from scipy import fft as sfft


class NavierStokes3D:
    """Unforced incompressible Navier-Stokes solver on a periodic cube.

    Uses Fourier pseudo-spectral discretization, 2/3 dealiasing,
    pressure projection, and Heun (RK2) time integration.
    Velocity components are ordered as (u, v, w).
    """

    def __init__(self, N=32, L=2 * np.pi, nu=1 / 400):
        """Initialize the spatial and Fourier grids."""
        if not isinstance(N, (int, np.integer)) or N < 8 or N % 2:
            raise ValueError("N must be an even integer >= 8")
        if not np.isfinite(L) or not np.isfinite(nu) or L <= 0 or nu < 0:
            raise ValueError("Require finite L > 0 and nu >= 0")

        self.N, self.L, self.nu = N, L, nu

        x = np.arange(N) * L / N
        self.X, self.Y, self.Z = np.meshgrid(
            x, x, x, indexing="ij"
        )

        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        kz = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)

        self.KX, self.KY, self.KZ = np.meshgrid(
            k, k, kz, indexing="ij"
        )
        self.Kvec = np.array([self.KX, self.KY, self.KZ])
        self.K2 = np.sum(self.Kvec**2, axis=0)

        self.K2inv = np.zeros_like(self.K2)
        np.divide(
            1.0, self.K2, out=self.K2inv, where=self.K2 > 0
        )

        # Filter integer Fourier modes independently of domain length.
        m = np.rint(np.fft.fftfreq(N) * N)
        mz = np.rint(np.fft.rfftfreq(N) * N)
        mx, my, mz = np.meshgrid(m, m, mz, indexing="ij")

        self.mask = (
            (np.abs(mx) < N / 3)
            & (np.abs(my) < N / 3)
            & (np.abs(mz) < N / 3)
        )

        self.u_hat = np.zeros(
            (3, N, N, N // 2 + 1), dtype=complex
        )

    def fft3(self, field):
        """Transform the three spatial axes to Fourier space."""
        return sfft.rfftn(field, axes=(-3, -2, -1))

    def ifft3(self, field_hat):
        """Recover a real field on the spatial grid."""
        return sfft.irfftn(
            field_hat,
            s=(self.N,) * 3,
            axes=(-3, -2, -1),
        )

    def project(self, field_hat):
        """Project a Fourier vector field onto divergence-free modes."""
        dot = np.sum(self.Kvec * field_hat, axis=0)
        return field_hat - self.Kvec * (dot * self.K2inv)

    def set_initial_velocity(self, velocity):
        """Set, filter, and project a real initial velocity field."""
        velocity = np.asarray(velocity)

        if velocity.shape != (3, self.N, self.N, self.N):
            raise ValueError(
                "Initial velocity must have shape (3, N, N, N)"
            )

        if not np.isrealobj(velocity) or not np.all(
            np.isfinite(velocity)
        ):
            raise ValueError("Initial velocity must be real and finite")

        self.u_hat = self.project(
            self.fft3(velocity.astype(float)) * self.mask
        )

    def set_taylor_green(self):
        """Initialize the classical three-dimensional Taylor-Green vortex."""
        x, y, z = (
            2 * np.pi / self.L * a
            for a in (self.X, self.Y, self.Z)
        )

        velocity = np.array(
            [
                np.sin(x) * np.cos(y) * np.cos(z),
                -np.cos(x) * np.sin(y) * np.cos(z),
                np.zeros_like(x),
            ]
        )

        self.set_initial_velocity(velocity)

    def curl_hat(self, u_hat):
        """Compute Fourier coefficients of the vorticity."""
        return 1j * np.array(
            [
                self.KY * u_hat[2] - self.KZ * u_hat[1],
                self.KZ * u_hat[0] - self.KX * u_hat[2],
                self.KX * u_hat[1] - self.KY * u_hat[0],
            ]
        )

    def nonlinear(self, u_hat):
        """Compute the dealiased and projected velocity-vorticity term."""
        u = self.ifft3(u_hat)
        omega = self.ifft3(self.curl_hat(u_hat))

        cross = np.cross(
            u, omega, axisa=0, axisb=0, axisc=0
        )

        return self.project(self.fft3(cross) * self.mask)

    def rhs(self, u_hat):
        """Evaluate the Navier-Stokes right-hand side in Fourier space."""
        return self.nonlinear(u_hat) - self.nu * self.K2 * u_hat

    def step(self, dt):
        """Advance one time step using Heun's explicit RK2 method."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        u_hat = self.u_hat
        k1 = self.rhs(u_hat)
        k2 = self.rhs(u_hat + dt * k1)

        self.u_hat = self.project(
            (u_hat + 0.5 * dt * (k1 + k2)) * self.mask
        )

    def velocity(self):
        """Return the velocity field with shape (3, N, N, N)."""
        return self.ifft3(self.u_hat)

    def vorticity(self):
        """Return the physical-space vorticity vector field."""
        return self.ifft3(self.curl_hat(self.u_hat))

    def kinetic_energy(self):
        """Return the domain-averaged kinetic energy per unit mass."""
        return 0.5 * np.mean(
            np.sum(self.velocity() ** 2, axis=0)
        )
