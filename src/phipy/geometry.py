from collections.abc import Callable, Sequence
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, splprep, splev
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import zscore

from phipy.array import identity


def normalized(array: npt.ArrayLike) -> np.ndarray:
    """Return vectors scaled to unit length along their last axis."""
    if np.ndim(array) >= 2:
        return np.asarray(array) / (np.linalg.norm(array, axis=-1)[:, None])
    else:
        return np.asarray(array) / np.linalg.norm(array, axis=-1)


def distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float | np.ndarray:
    """Return Euclidean distance between points or corresponding point arrays."""
    a = np.asarray(a)
    b = np.asarray(b)

    if np.ndim(a) == np.ndim(b) == 1:
        return np.linalg.norm(b - a)
    else:
        return np.linalg.norm(b - a, axis=1)


def direction_vector(start_point: npt.ArrayLike, end_point: npt.ArrayLike) -> np.ndarray:
    """Return the unit vector from one point to another."""
    return normalized(end_point - start_point)


def closest_point(array: npt.ArrayLike, point: npt.ArrayLike) -> np.ndarray:
    """Return the closest point in an array; point and array may be swapped."""
    array = np.asarray(array)
    point = np.asarray(point)

    if array.ndim == 1 and point.ndim == 2:
        array, point = point, array

    return array[closest_index(array, point)]


def closest_distance(x1: npt.ArrayLike, x2: npt.ArrayLike) -> float:
    """Return the smallest pairwise distance between two point sets."""
    x1 = np.atleast_2d(np.asarray(x1, dtype=float))
    x2 = np.atleast_2d(np.asarray(x2, dtype=float))
    return cdist(x1, x2).min()


def closest_index(
    array: npt.ArrayLike,
    coordinate: npt.ArrayLike,
    apply: Sequence[int] | None = None,
) -> int | float:
    """Return the index of the array element closest to a coordinate."""
    if np.isnan(coordinate).any():
        return np.nan

    array = np.asarray(array)
    coordinate = np.asarray(coordinate)

    if np.ndim(array) == 1:
        index = np.argmin(np.abs(array - coordinate))
    else:
        index = np.argmin(np.linalg.norm(array - coordinate, axis=-1))

    if apply is not None:
        return apply[index]
    else:
        return index


def close_points(
    array: npt.ArrayLike,
    point: npt.ArrayLike | float = 0,
    delta: float | None = None,
) -> np.ndarray:
    """Return a mask for points within a distance threshold of a point."""
    array = np.asarray(array)
    if np.ndim(array) == 1:
        array = array[:, None]

    distances = np.linalg.norm(array - point, axis=-1)

    if delta is None:
        delta = np.ptp(distances) * 0.01

    return np.abs(distances) < delta


def rotation_matrix(
    angle: float,
    direction_vector: npt.ArrayLike | None = None,
    degrees: bool = True,
) -> np.ndarray:
    """Return a 2D rotation matrix, or a 3D rotation around an axis."""
    if degrees:
        angle = np.deg2rad(angle)

    if direction_vector is None:
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    else:
        direction_vector = normalized(direction_vector)
        matrix = Rotation.from_rotvec(angle * direction_vector)
        return matrix.as_matrix()


@overload
def align_vectors(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    return_angle: Literal[False] = False,
) -> np.ndarray: ...


@overload
def align_vectors(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    return_angle: Literal[True],
) -> tuple[np.ndarray, float]: ...


def align_vectors(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    return_angle: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Return a 4x4 transform that rotates vector a onto vector b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("vectors must be (3,)")

    rotation, _ = Rotation.align_vectors([b], [a])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()

    if return_angle:
        angle = vector_angle(a, b, degrees=False)
        return matrix, angle

    return matrix


def vector_angle(
    v1: npt.ArrayLike,
    v2: npt.ArrayLike,
    plane: Literal["xy", "yz", "xz"] | None = None,
    degrees: bool = True,
    signed: bool | Literal["xy", "yz", "xz"] = False,
) -> float:
    """Return the angle between two vectors, optionally projected or signed."""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    axes = None

    if len(v1) == 2:
        v1 = np.pad(v1, [0, 1])
        v2 = np.pad(v2, [0, 1])
        plane = "xy"

    if plane == "xy":
        axes = [0, 1]
    elif plane == "yz":
        axes = [1, 2]
    elif plane == "xz":
        axes = [0, 2]

    if plane and axes is not None:
        v1, v2 = v1[axes], v2[axes]

    if signed:
        if axes is None:
            if signed not in ("xy", "yz", "xz"):
                raise ValueError(
                    "For 3D angles, `signed` must indicate a projection axis for the sign."
                )
            sign = np.sign(vector_angle(v1, v2, plane=signed))
        else:
            sign = np.sign(np.cross(np.pad(v1, [0, 1]), np.pad(v2, [0, 1]))[-1])
    else:
        sign = 1

    conversion = np.rad2deg if degrees else identity

    return conversion(
        sign * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    )


def circle_segment(
    origin: npt.ArrayLike,
    d1: npt.ArrayLike,
    d2: npt.ArrayLike,
    radius: float = 1,
) -> np.ndarray:
    """Return points along the circular arc between two directions."""
    origin = np.asarray(origin)
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)
    dot_product = np.clip(np.dot(d1_norm, d2_norm), -1.0, 1.0)
    angle = np.arccos(dot_product)
    normal = np.cross(d1_norm, d2_norm)
    normal = normal / np.linalg.norm(normal)

    n_points = 50
    t = np.linspace(0, angle, n_points)

    points = np.zeros((n_points, 3))
    for i, ti in enumerate(t):
        cos_t = np.cos(ti)
        sin_t = np.sin(ti)
        points[i] = (
            d1_norm * cos_t
            + np.cross(normal, d1_norm) * sin_t
            + normal * np.dot(normal, d1_norm) * (1 - cos_t)
        )

    points = origin + radius * points
    return points


def create_line(
    point1: npt.ArrayLike,
    point2: npt.ArrayLike,
    stepsize: float = 0.1,
) -> np.ndarray:
    """Return evenly spaced points from point1 toward point2."""
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    distance = np.linalg.norm(point2 - point1)
    n_points = int(distance / stepsize) + 1
    direction = (point2 - point1) / distance
    steps = np.arange(n_points) * stepsize
    return point1 + direction[None, :] * steps[:, None]


def line_length(array: npt.ArrayLike) -> float:
    """Return the total length of a polyline."""
    array = np.asarray(array)
    return np.linalg.norm(np.diff(array, axis=0), axis=-1).sum()


def order_coordinates(
    coords: npt.ArrayLike,
    approx_start_point: npt.ArrayLike = (0, 0, 0),
    filter: bool = True,
) -> np.ndarray:
    """Order unordered coordinates into a nearest-neighbor polyline."""
    coords = [list(c) for c in coords]

    distance_sums = []
    for i in range(len(coords)):
        sum_distances = sum(
            distance(coords[i], coords[j]) for j in range(len(coords)) if i != j
        )
        distance_sums.append(sum_distances)

    endpoints = sorted(
        range(len(distance_sums)), key=lambda k: distance_sums[k], reverse=True
    )[:2]

    endpoint_coords = [coords[endpoints[0]], coords[endpoints[1]]]
    start_point = min(endpoint_coords, key=lambda p: distance(p, approx_start_point))

    ordered_coords = [start_point]
    coords.remove(start_point)

    while coords:
        last_coord = ordered_coords[-1]
        next_coord = min(coords, key=lambda coord: distance(last_coord, coord))
        ordered_coords.append(next_coord)
        coords.remove(next_coord)

    points = np.array(ordered_coords)
    if filter:
        good = ~(zscore(np.linalg.norm(np.diff(points, axis=0), axis=-1)) > 5)
        points = points[np.pad(good, (0, 1)) & np.pad(good, (1, 0))]

    return points


def smooth_centerline(
    points: npt.ArrayLike,
    start: float = 0,
    end: float = 1,
    n_points: int = 100,
    s: float = 0.01,
) -> np.ndarray:
    """Fit and sample a smooth spline through centerline points."""
    points = np.asarray(points)
    tck, u = splprep(points.T, s=s)
    u_new = np.linspace(start, end, n_points)
    smoothed_points = np.array(splev(u_new, tck, der=0)).T
    return smoothed_points


def generate_smooth_points(
    vertices: npt.ArrayLike,
    num_points: int = 100,
    window_size: int = 5,
) -> np.ndarray:
    """Resample and smooth a closed sequence of vertices."""
    vertices = np.asarray(vertices)
    num_points *= 3
    vertices = np.concatenate([vertices, vertices, vertices])
    distances = np.cumsum(np.sqrt(np.sum(np.diff(vertices, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)

    uniform_distances = np.linspace(0, distances[-1], num_points)
    interpolator = interp1d(distances, vertices, axis=0, kind="linear")
    uniform_points = interpolator(uniform_distances)

    smoothed_points = np.zeros_like(uniform_points)
    for i in range(num_points):
        start = max(0, i - window_size // 2)
        end = min(num_points, i + window_size // 2 + 1)
        smoothed_points[i] = np.mean(uniform_points[start:end], axis=0)

    return smoothed_points[num_points // 3 : num_points // 3 * 2 + 1]


def max_distance_subset(
    coords: npt.ArrayLike,
    k: int,
    mask: bool = False,
    tqdm: Callable[[range], Sequence[int]] = identity,
) -> np.ndarray | list[bool]:
    """Select k points by greedy farthest-point sampling."""
    coords = np.asarray(coords)
    n = len(coords)
    if k > n:
        raise ValueError("k must be less than or equal to the number of points")

    subset = [coords[np.random.randint(n)]]
    distances = np.linalg.norm(coords - subset[0], axis=1)

    for _ in tqdm(range(1, k)):
        next_point_index = np.argmax(distances)
        next_point = coords[next_point_index]
        subset.append(next_point)
        new_distances = np.linalg.norm(coords - next_point, axis=1)
        distances = np.minimum(distances, new_distances)

    if mask:
        idxs = [np.where(coords == m)[0][0] for m in subset]
        mask = [(True if i in idxs else False) for i in range(len(coords))]
        return mask

    return np.array(subset)


def cartesian_to_cylindrical(
    x: npt.ArrayLike | float,
    y: npt.ArrayLike | float,
    z: npt.ArrayLike | float | None = None,
) -> np.ndarray:
    """Convert Cartesian coordinates to cylindrical coordinates."""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if z is None:
        return np.array([r, phi])
    else:
        return np.array([r, phi, z])


def cartesian_to_polar(x: npt.ArrayLike | float, y: npt.ArrayLike | float) -> np.ndarray:
    """Convert 2D Cartesian coordinates to polar coordinates."""
    return np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)])


def polar_to_cartesian(
    r: npt.ArrayLike | float,
    phi: npt.ArrayLike | float,
) -> np.ndarray:
    """Convert polar coordinates to 2D Cartesian coordinates."""
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def cartesian_to_spherical(
    x: npt.ArrayLike | float,
    y: npt.ArrayLike | float,
    z: npt.ArrayLike | float,
    degrees: bool = False,
) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if degrees:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
    return np.array([r, theta, phi])


def spherical_to_cartesian(
    r: npt.ArrayLike | float,
    theta: npt.ArrayLike | float,
    phi: npt.ArrayLike | float,
    degrees: bool = False,
) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates."""
    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z])
