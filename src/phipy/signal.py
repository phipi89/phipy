from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import zscore

from phipy.array import linspace_like


def cleaned_signal(
    signal: npt.ArrayLike,
    carrier: npt.ArrayLike | None = None,
    threshold: float = 5,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Remove signal samples whose absolute z-score exceeds a threshold."""
    signal = np.asarray(signal)
    z_scores = np.abs(zscore(signal))
    selection = z_scores < threshold

    if carrier is None:
        return signal[selection]
    else:
        carrier = np.asarray(carrier)
        return carrier[selection], signal[selection]


def moving_window(
    x_vals: npt.ArrayLike,
    y_vals: npt.ArrayLike,
    window: float = 5,
    function: Callable[..., Any] = np.mean,
    support: int | npt.ArrayLike = 250,
    nan_policy: str = "raise",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a function to y-values within a moving x-window."""
    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)

    if np.isnan([x_vals, y_vals]).any():
        if nan_policy == "raise":
            raise ValueError(
                'Input data contains NaNs. Consider setting `nan_policy`="remove".'
            )

        nans = np.isnan([x_vals, y_vals]).any(axis=0)
        x_vals = x_vals[~nans]
        y_vals = y_vals[~nans]

    if isinstance(support, int):
        support = linspace_like(x_vals, support)
    applied = [
        function(y_vals[np.abs(x_vals - t) <= window], **kwargs) for t in support
    ]
    applied = np.asarray(applied)

    return support, applied
