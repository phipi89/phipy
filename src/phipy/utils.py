from collections.abc import Callable
from numbers import Integral
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


def try_else_nan(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any | float:
    """Return a function result, or NaN if the call raises an exception."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return np.nan


def identity(arg: T) -> T:
    """Return the input unchanged."""
    return arg


def map_to_interval(
    x: npt.ArrayLike,
    low: float = 0,
    high: float = 1,
    sine: bool = False,
) -> np.ndarray:
    """Map values from [low, high] to [0, 1], clamped outside."""
    if low == high:
        raise ValueError("low and high must be different.")

    y = np.clip((np.asarray(x) - low) / (high - low), 0, 1)
    if sine:
        y = 0.5 - 0.5 * np.cos(np.pi * y)
    return y


def ramp(
    x: int | npt.ArrayLike,
    low: float = 0,
    high: float = 1,
    sine: bool = False,
    bounce: bool = False,
) -> np.ndarray:
    """Return a ramp with x samples, or x.size samples for array-like inputs."""
    n = int(x) if isinstance(x, Integral) else np.size(x)
    if bounce:
        up_count = (n + 1) // 2
        down_count = n - up_count
        y = np.linspace(0, 1, up_count)
        if down_count:
            if n % 2 == 0:
                y = np.concatenate([y, np.linspace(1, 0, down_count)])
            else:
                y = np.concatenate([y, np.linspace(1, 0, down_count + 1)[1:]])
    else:
        y = np.linspace(0, 1, n)
    if sine:
        y = 0.5 - 0.5 * np.cos(np.pi * y)
    return low + y * (high - low)
