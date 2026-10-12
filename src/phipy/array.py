from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


def normalized(array: npt.ArrayLike) -> np.ndarray:
    """Return vectors scaled to unit length along their last axis."""
    if np.ndim(array) >= 2:
        return np.asarray(array) / (np.linalg.norm(array, axis=-1)[:, None])
    else:
        return np.asarray(array) / np.linalg.norm(array, axis=-1)


def close_loop(arr: npt.ArrayLike, axis: int = 0) -> np.ndarray:
    """Return an array with its first element appended along an axis."""
    return np.concatenate([arr, np.take(arr, [0], axis=axis)], axis=axis)


def drop_nan(array: npt.ArrayLike) -> np.ndarray:
    """Return an array with NaN values or NaN-containing rows removed."""
    array = np.asarray(array)
    if np.ndim(array) == 1:
        return array[~np.isnan(array)]

    mask = np.isnan(array).any(axis=-1)
    return array[~mask]


def without(array: npt.ArrayLike, values: npt.ArrayLike) -> np.ndarray:
    """Return entries from an array that are not in values, preserving order."""
    array = np.asarray(array)
    values = np.asarray(values)

    if array.ndim == 1:
        return array[~np.isin(array, values)]

    values = np.atleast_2d(values)
    if values.shape[-1] != array.shape[-1]:
        raise ValueError("values must have the same trailing dimension as array.")

    mask = (array[:, None] == values[None]).all(axis=-1).any(axis=1)
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


def center_crop(
    array: npt.ArrayLike,
    shape: int | tuple[int, ...],
    fill_value: Any | None = None,
) -> np.ndarray:
    """Return a centered crop, optionally padding to the requested shape."""
    array = np.asarray(array)
    if isinstance(shape, int):
        shape = (shape,)

    if len(shape) != array.ndim:
        raise ValueError("shape must have one entry per array dimension.")
    if any(size < 0 for size in shape):
        raise ValueError("shape entries must be non-negative.")

    crop_slices = []
    insert_slices = []
    needs_padding = False
    for current, target in zip(array.shape, shape):
        crop_size = min(current, target)
        crop_start = (current - crop_size) // 2
        insert_start = (target - crop_size) // 2
        crop_slices.append(slice(crop_start, crop_start + crop_size))
        insert_slices.append(slice(insert_start, insert_start + crop_size))
        needs_padding |= target > current

    cropped = array[tuple(crop_slices)]
    if not needs_padding:
        return cropped
    if fill_value is None:
        raise ValueError(
            "fill_value is required when shape is larger than array.shape."
        )

    result = np.full(shape, fill_value, dtype=array.dtype)
    result[tuple(insert_slices)] = cropped
    return result


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
