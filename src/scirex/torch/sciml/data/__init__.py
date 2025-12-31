"""
Data generation utilities for FNO models.

This package provides tools for generating synthetic datasets for training
Fourier Neural Operator models on various PDE problems.
"""

from .generate_sr import DataGenerator

__all__ = ['DataGenerator']
