import unittest
import warnings

import numpy as np

from phipy.geometry import plane_from_points


class PlaneFromPointsTests(unittest.TestCase):
    def test_plane_from_three_points(self):
        points = [[0, 0, 2], [1, 0, 2], [0, 1, 2]]

        center, normal = plane_from_points(points)

        np.testing.assert_allclose(center, [1 / 3, 1 / 3, 2])
        np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-12)

    def test_overdetermined_plane_warns_and_uses_least_squares(self):
        points = np.array(
            [
                [-1, -1, 2.01],
                [-1, 1, -4.01],
                [1, -1, 5.99],
                [1, 1, 0.01],
                [0, 0, 1.0],
            ]
        )

        with self.assertWarnsRegex(UserWarning, "least-squares"):
            center, normal = plane_from_points(points)

        expected_normal = np.array([-2, 3, 1]) / np.sqrt(14)
        np.testing.assert_allclose(center, points.mean(axis=0))
        np.testing.assert_allclose(normal, expected_normal, atol=2e-3)
        self.assertAlmostEqual(np.linalg.norm(normal), 1)

    def test_warning_can_be_suppressed(self):
        points = [[0, 0], [1, 0], [2, 0]]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            center, normal = plane_from_points(points, warn=False)

        self.assertEqual(caught, [])
        np.testing.assert_allclose(center, [1, 0])
        np.testing.assert_allclose(normal, [0, 1])

    def test_supports_fully_determined_hyperplanes(self):
        points = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        center, normal = plane_from_points(points)

        np.testing.assert_allclose(center, [0.25, 0.25, 0.25, 0])
        np.testing.assert_allclose(normal, [0, 0, 0, 1], atol=1e-12)

    def test_normal_is_canonical_and_order_independent(self):
        points = np.array([[0, 0, 1], [1, 0, 3], [0, 1, -2], [1, 1, 0], [2, -1, 8]])

        _, normal = plane_from_points(points, warn=False)
        _, permuted_normal = plane_from_points(points[::-1], warn=False)

        np.testing.assert_allclose(normal, permuted_normal, atol=1e-12)
        self.assertGreater(normal[np.argmax(np.abs(normal))], 0)

    def test_rejects_invalid_or_degenerate_points(self):
        invalid_points = (
            [1, 2, 3],
            [[0], [1]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0], [np.nan, 1]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        )

        for points in invalid_points:
            with self.subTest(points=points), self.assertRaises(ValueError):
                plane_from_points(points)


if __name__ == "__main__":
    unittest.main()
