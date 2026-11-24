import numpy as np


class NavierStokes2D:
    """Unforced 2D incompressible Navier-Stokes on a periodic square.

    Uses vorticity-streamfunction formulation, Fourier derivatives,
    2/3 dealiasing, and classical RK4. Mean velocity is fixed to zero.
    """

    def __init__(self, N=64, L=2 * np.pi, nu=0.01):
        if N < 8 or N % 2:
            raise ValueError("N must be an even integer >= 8")
        if L <= 0 or nu < 0:
            raise ValueError("Require L > 0 and nu >= 0")
        self.N, self.L, self.nu = N, L, nu
        x = np.arange(N) * L / N
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2_safe = self.K2.copy()
        self.K2_safe[0, 0] = 1.0
        modes = np.fft.fftfreq(N) * N
        MX, MY = np.meshgrid(modes, modes, indexing="ij")
        self.mask = (np.abs(MX) < N / 3) & (np.abs(MY) < N / 3)
        self.omega_hat = np.zeros((N, N), dtype=complex)

    def set_initial_vorticity(self, omega):
        """
        Set the initial vorticity field of the solver.

        Transform the physical-space vorticity field into Fourier
        space, apply the 2/3 dealiasing rule, and remove the
        zero-wavenumber mode.

        Parameters
        ----------
        omega : array_like
            Initial vorticity field in physical space.
            Must have shape (N, N), where N is the number
            of grid points in each spatial direction.

        Returns
        -------
        None
            Updates the internal Fourier-space vorticity field.

        Raises
        ------
        ValueError
            If omega has an incorrect shape or contains values
            that cannot be converted to floats.
        TypeError
            If omega cannot be converted to a floating-point array.

        Examples
        --------
        >>> import numpy as np
        >>> from src.ns2d import NavierStokes2D
        >>> solver = NavierStokes2D(N=32)
        >>> omega = 2 * np.sin(solver.X) * np.sin(solver.Y)
        >>> solver.set_initial_vorticity(omega)
        """
        omega = np.asarray(omega, dtype=float)
        if omega.shape != (self.N, self.N):
            raise ValueError("Initial vorticity must have shape (N, N)")
        self.omega_hat = np.fft.fft2(omega) * self.mask
        self.omega_hat[0, 0] = 0.0

    def velocity_from_hat(self, omega_hat):
        # omega = -laplacian(psi), so psi_hat = omega_hat / |k|^2
        psi_hat = omega_hat / self.K2_safe
        u = np.fft.ifft2(1j * self.KY * psi_hat).real
        v = np.fft.ifft2(-1j * self.KX * psi_hat).real
        return u, v

    def velocity(self):
        return self.velocity_from_hat(self.omega_hat)

    def vorticity(self):
        return np.fft.ifft2(self.omega_hat).real

    def rhs(self, omega_hat):
        """
        Compute the right-hand side of the vorticity equation.

        Evaluate the nonlinear advection and viscous diffusion
        terms of the two-dimensional incompressible Navier-Stokes
        equations in Fourier space. Apply the 2/3 dealiasing rule
        to remove unresolved high-wavenumber modes.

        Parameters
        ----------
        omega_hat : numpy.ndarray
            Complex Fourier coefficients of the vorticity field.
            Must have shape (N, N), where N is the number of
            grid points in each spatial direction.

        Returns
        -------
        numpy.ndarray
            Complex Fourier coefficients of the time derivative
            of vorticity, with shape (N, N).

        Raises
        ------
        ValueError
            If the input shape is incompatible with the solver's
            spectral arrays or Fourier transform operations.
        TypeError
            If the input does not support the required numerical
            operations.

        Notes
        -----
        The nonlinear advection term is evaluated in physical
        space and transformed back into Fourier space.
        The viscous diffusion term is evaluated directly in
        Fourier space.

        Examples
        --------
        >>> import numpy as np
        >>> from src.ns2d import NavierStokes2D
        >>> solver = NavierStokes2D(N=32, nu=0.01)
        >>> omega = 2 * np.sin(solver.X) * np.sin(solver.Y)
        >>> solver.set_initial_vorticity(omega)
        >>> domega_hat = solver.rhs(solver.omega_hat)
        >>> domega_hat.shape
        (32, 32)
        """
        u, v = self.velocity_from_hat(omega_hat)
        wx = np.fft.ifft2(1j * self.KX * omega_hat).real
        wy = np.fft.ifft2(1j * self.KY * omega_hat).real
        nonlinear_hat = np.fft.fft2(u * wx + v * wy) * self.mask
        return (-nonlinear_hat - self.nu * self.K2 * omega_hat) * self.mask

    def step(self, dt):
        """
        Advance the vorticity field by one time step.

        Integrate the Fourier-space vorticity equation using
        the classical fourth-order Runge-Kutta (RK4) method.
        Apply the dealiasing mask and remove the zero mode
        after the update.

        Parameters
        ----------
        dt : float
            Positive time step size. The caller must choose
            a sufficiently small value for numerical stability.

        Returns
        -------
        None
            Updates the internal Fourier-space vorticity field.

        Raises
        ------
        ValueError
            If dt is less than or equal to zero.
        TypeError
            If dt is not a compatible numeric value.

        Examples
        --------
        >>> import numpy as np
        >>> from src.ns2d import NavierStokes2D
        >>> solver = NavierStokes2D(N=32, nu=0.01)
        >>> omega = 2 * np.sin(solver.X) * np.sin(solver.Y)
        >>> solver.set_initial_vorticity(omega)
        >>> solver.step(0.001)
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        w = self.omega_hat
        k1 = self.rhs(w)
        k2 = self.rhs(w + dt * k1 / 2)
        k3 = self.rhs(w + dt * k2 / 2)
        k4 = self.rhs(w + dt * k3)
        self.omega_hat = (w + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6) * self.mask
        self.omega_hat[0, 0] = 0.0

    def kinetic_energy(self):
        u, v = self.velocity()
        return 0.5 * np.mean(u * u + v * v)
