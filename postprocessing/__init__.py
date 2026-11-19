"""
Post-processing tools for the turbulence-simulation project.

Currently provides a single entry point::

    from postprocessing import plot_turbulence_budget

which computes and plots the resolved kinetic-energy budget (production,
transport, dissipation and a global closure diagnostic) from a saved 3D
velocity field on a triply periodic grid.
"""

from .turbulence_budget import plot_turbulence_budget

__all__ = ["plot_turbulence_budget"]

__version__ = "0.1.0"
