from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation

from .array import normalized
from .geometry import vector_angle


def align_along_vector(
    coords, vector, up=np.array([0, 0, 1.0]), shift=np.array([0, 0, 0.0])
):
    """Rotate and translate a set of coordinates such that the z-axis
    aligns with `vector`.
    """

    if len(vector) == 2:
        shift += vector[0]
        vector = normalized(vector[1] - vector[0])

    rotvec = normalized(np.cross(up, vector)) * vector_angle(up, vector, degrees=False)
    m = Rotation.from_rotvec(rotvec).as_matrix()

    coords = coords @ m.T
    coords += shift
    return coords


def rotate_around(array, axis, angle, deg=False):
    """Rotate 3D coordinates around an oriented axis.

    ``axis`` may be a direction vector through the origin with shape ``(3,)``,
    or two points on the axis with shape ``(2, 3)``. Angles are in radians
    unless ``deg`` is true.
    """
    array = np.asarray(array)
    axis = np.asarray(axis, dtype=float)

    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError("array must have shape (..., 3)")

    if axis.shape == (3,):
        origin = np.zeros(3)
        direction = axis
    elif axis.shape == (2, 3):
        origin = axis[0]
        direction = axis[1] - origin
    else:
        raise ValueError("axis must have shape (3,) or (2, 3)")

    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("axis direction must be nonzero")

    rotation = Rotation.from_rotvec(angle * direction / length, degrees=deg)
    return (array - origin) @ rotation.as_matrix().T + origin


def apply_affine(array, affine_matrix, inv=False):
    """Apply an affine transform to `array`."""

    array = np.asarray(array)
    n = np.asarray(array).shape[-1]
    array = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, 1)], constant_values=1)

    if inv:
        affine_matrix = np.linalg.inv(affine_matrix)

    transform = partial(np.dot, affine_matrix)
    array = np.apply_along_axis(transform, -1, array)[..., :n]
    return array


def affine_transform_coords(coords, transformation_matrix):
    # TODO: Check if this is the same as apply_affine
    #       I fear that it is not.
    coords = np.asarray(coords)
    points_homogeneous = np.hstack([coords, np.ones((coords.shape[0], 1))])
    return np.dot(points_homogeneous, transformation_matrix.T)


def as_affine(R, t):
    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def invert_affine(M):
    """
    Invert an affine matrix in
    of arbitryry dimension.
    """

    n = M.shape[0]
    A = M[:, :n]
    t = M[:, n]
    A_inv = np.linalg.inv(A)
    t_inv = -A_inv @ t
    return np.hstack([A_inv, t_inv.reshape(n, 1)])


invert_affine_matrix = invert_affine
