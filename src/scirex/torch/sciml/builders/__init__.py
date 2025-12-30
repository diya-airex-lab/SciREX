"""
PDE solvers and builders for FNO models.

This package provides builder classes for different PDE types that can be used
to generate synthetic training data for FNO super-resolution models.
"""

from .base import BaseSolver
from .wave import WaveSolver
from .heat import HeatSolver
from .advection import AdvectionSolver

__all__ = [
    'BaseSolver',
    'WaveSolver',
    'HeatSolver',
    'AdvectionSolver',
]
