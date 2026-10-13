from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.ns2d import NavierStokes2D


def main():
    solver = NavierStokes2D(N=64, nu=0.01)
    # Two-dimensional smooth vortical initial condition, not fully developed turbulence.
    omega0 = (2 * np.sin(solver.X) * np.sin(solver.Y)
              + 0.6 * np.cos(3 * solver.X + 2 * solver.Y)
              + 0.4 * np.sin(2 * solver.X - 3 * solver.Y))
    solver.set_initial_vorticity(omega0)
    dt, steps = 0.005, 200
    for n in range(steps):
        solver.step(dt)
        if (n + 1) % 50 == 0:
            print(f't={(n+1)*dt:.3f}, kinetic energy={solver.kinetic_energy():.6f}')
    u, v = solver.velocity()
    speed = np.sqrt(u*u + v*v)
    out = Path('results/figures')
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    extent = (0, solver.L, 0, solver.L)
    im1 = axes[0].imshow(speed.T, origin='lower', extent=extent)
    axes[0].set_title('Velocity magnitude')
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(solver.vorticity().T, origin='lower', extent=extent, cmap='RdBu_r')
    axes[1].set_title('Vorticity')
    fig.colorbar(im2, ax=axes[1])
    for ax in axes:
        ax.set(xlabel='x', ylabel='y')
    path = out / 'ns2d_final.png'
    fig.savefig(path, dpi=160)
    print(f'Saved {path}')
    plt.show()


if __name__ == '__main__':
    main()
