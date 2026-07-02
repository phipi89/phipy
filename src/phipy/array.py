from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


def close_loop(arr: npt.ArrayLike, axis: int = 0) -> np.ndarray:
    """Return an array with its first element appended along an axis."""
    return np.concatenate([arr, np.take(arr, [0], axis=axis)], axis=axis)


def try_else_nan(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any | float:
    """Return a function result, or NaN if the call raises an exception."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return np.nan


def identity(arg: T) -> T:
    """Return the input unchanged."""
    return arg


def drop_nan(array: npt.ArrayLike) -> np.ndarray:
    """Return an array with NaN values or NaN-containing rows removed."""
    array = np.asarray(array)
    if np.ndim(array) == 1:
        return array[~np.isnan(array)]

    mask = np.isnan(array).any(axis=-1)
    return array[~mask]


def fill_nan(array: npt.ArrayLike) -> np.ndarray:
    """Return an array with NaNs filled by linear interpolation."""
    array = np.asarray(array, dtype=float).copy()
    if np.ndim(array) == 2:
        x, y = array.T
        return np.array([fill_nan(x), fill_nan(y)]).T

    mask = np.isnan(array)
    array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
    return array


def scale_to_unit_interval(array: npt.ArrayLike) -> np.ndarray:
    """Scale values linearly to the interval [0, 1]."""
    array = np.asarray(array)
    min_val = np.min(array)
    max_val = np.max(array)
    if min_val == max_val:
        raise ValueError("Cannot scale an array with all elements equal.")
    return (array - min_val) / (max_val - min_val)


def linspace_like(
    array: npt.ArrayLike,
    n: int = 100,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Return evenly spaced values spanning an array's range by default."""
    array = np.asarray(array)
    if start is None:
        start = np.min(array)
    if end is None:
        end = np.max(array)
    return np.linspace(start, end, n)


def shorten(array: npt.ArrayLike, n: int = 1) -> np.ndarray:
    """Return adjacent values averaged over an offset of n samples."""
    array = np.asarray(array)
    return (array[n:] + array[:-n]) / (n + 1)
