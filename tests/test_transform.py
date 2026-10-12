import unittest

import numpy as np

from phipy.transform import rotate_around


class RotateAroundTests(unittest.TestCase):
    def test_rotates_around_direction_in_radians(self):
        result = rotate_around([1, 0, 0], [0, 0, 1], np.pi / 2)

        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-12)

    def test_rotates_around_translated_axis_in_degrees(self):
        axis = [[1, 0, 0], [1, 0, 1]]
        points = np.array([[2, 0, 0], [1, 0, 4]])

        result = rotate_around(points, axis, 90, deg=True)

        np.testing.assert_allclose(result, [[1, 1, 0], [1, 0, 4]], atol=1e-12)

    def test_preserves_leading_dimensions(self):
        points = np.array([[[1, 0, 0], [0, 1, 0]]])

        result = rotate_around(points, [0, 0, 1], np.pi)

        self.assertEqual(result.shape, points.shape)
        np.testing.assert_allclose(result, [[[-1, 0, 0], [0, -1, 0]]], atol=1e-12)

    def test_reversing_axis_reverses_rotation(self):
        forward = rotate_around([1, 0, 0], [[0, 0, 0], [0, 0, 1]], 90, deg=True)
        reverse = rotate_around([1, 0, 0], [[0, 0, 1], [0, 0, 0]], 90, deg=True)

        np.testing.assert_allclose(forward, [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(reverse, [0, -1, 0], atol=1e-12)

    def test_rejects_invalid_shapes_and_zero_length_axes(self):
        with self.assertRaisesRegex(ValueError, "array must have shape"):
            rotate_around([1, 0], [0, 0, 1], 0)
        with self.assertRaisesRegex(ValueError, "axis must have shape"):
            rotate_around([1, 0, 0], [0, 1], 0)
        with self.assertRaisesRegex(ValueError, "axis direction must be nonzero"):
            rotate_around([1, 0, 0], [[1, 1, 1], [1, 1, 1]], 0)


if __name__ == "__main__":
    unittest.main()
