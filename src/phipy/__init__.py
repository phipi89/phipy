"""Convenience namespace for phipy submodules and common helpers."""

from . import array, full, geometry, images, mesh, notebook, signal
from .array import identity
from .geometry import normalized

normalised = normalized

__all__ = [
    "array",
    "full",
    "geometry",
    "identity",
    "images",
    "mesh",
    "normalised",
    "normalized",
    "notebook",
    "signal",
]
