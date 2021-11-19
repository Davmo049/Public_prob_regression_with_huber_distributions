import unittest
import numpy as np
from . import covers

class TestGeometricMedian(unittest.TestCase):
    def test_trivial(self):
        points = np.array([[1.0], [0.0], [-1.0]])
        ret = covers.geometric_median(points)

        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(len(ret.shape), 1)
        self.assertEqual(ret.shape[0], 1)
        self.assertAlmostEqual(ret[0], 0.0)

    def test_2_d_example(self):
        points = np.array([[-1.0, 0], [1.0, 0], [0.0, 1]])
        ret = covers.geometric_median(points)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(len(ret.shape), 1)
        self.assertEqual(ret.shape[0], 2)
        self.assertAlmostEqual(ret[0], 0.0)
        self.assertAlmostEqual(ret[1], 1/np.sqrt(3), places=3)

    def test_2_d_weighted(self):
        points = np.array([[-1.0, 0], [1.0, 0], [0.0, 1], [0.0, -1]])
        weights = np.array([1.0, 1, 2, 1])
        ret = covers.geometric_median(points,weights)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(len(ret.shape), 1)
        self.assertEqual(ret.shape[0], 2)
        self.assertAlmostEqual(ret[0], 0.0)
        self.assertAlmostEqual(ret[1], 1/np.sqrt(3), places=3)

    def test_circle(self):
        points = []
        for i in range(10):
            th = i*np.pi*2/10
            points.append((np.sin(th)+1.2, np.cos(th)+3.3))
        points = np.array(points)
        ret = covers.geometric_median(points)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(len(ret.shape), 1)
        self.assertEqual(ret.shape[0], 2)
        self.assertAlmostEqual(ret[0], 1.2, places=3)
        self.assertAlmostEqual(ret[1], 3.3, places=3)

# class TestWelzl(unittest.TestCase):
#     def test_empty(self):
#         points = np.empty((0,2))
#         center, radius = covers.minimum_covering_sphere(points)
#         self.assertEqual(radius, 0.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
# 
#     def test_one_element(self):
#         points = np.array([[3.3,5.5]])
#         center, radius = covers.minimum_covering_sphere(points)
#         self.assertEqual(radius, 0.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertEqual(center[0], 3.3)
#         self.assertEqual(center[1], 5.5)
# 
#     def test_two_element(self):
#         points = np.array([[0.2, 3.3], [2.2, 3.3]])
#         center, radius = covers.minimum_covering_sphere(points)
#         self.assertAlmostEqual(radius, 1.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertAlmostEqual(center[0], 1.2)
#         self.assertAlmostEqual(center[1], 3.3)
# 
#     def test_three_element(self):
#         points = np.array([[-np.sqrt(3)/2, 1/2], [-np.sqrt(3)/2, -1/2], [1, 0.0]])
#         center, radius = covers.minimum_covering_sphere(points)
#         self.assertAlmostEqual(radius, 1.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertAlmostEqual(center[0], 0.0)
#         self.assertAlmostEqual(center[1], 0.0)
# 
#     def test_many_element(self):
#         num_points = 500
#         points = np.random.normal(size=(num_points, 2))
#         points /= np.linalg.norm(points, axis=1).reshape(num_points, 1)
#         points[0] = np.array([-9, 0])
#         points[-1] = np.array([11, 0])
#         center, radius = covers.minimum_covering_sphere(points)
#         self.assertAlmostEqual(radius, 10.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertAlmostEqual(center[0], 1)
#         self.assertAlmostEqual(center[1], 0)
# 
# 
# class TestWelzlIterative(unittest.TestCase):
#     def test_empty(self):
#         points = np.empty((0,2))
#         center, radius = covers.minimum_covering_sphere_iterative(points)
#         self.assertEqual(radius, 0.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
# 
#     def test_one_element(self):
#         points = np.array([[3.3,5.5]])
#         center, radius = covers.minimum_covering_sphere_iterative(points)
#         self.assertEqual(radius, 0.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertEqual(center[0], 3.3)
#         self.assertEqual(center[1], 5.5)
# 
#     def test_two_element(self):
#         points = np.array([[0.2, 3.3], [2.2, 3.3]])
#         center, radius = covers.minimum_covering_sphere_iterative(points)
#         self.assertAlmostEqual(radius, 1.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertAlmostEqual(center[0], 1.2)
#         self.assertAlmostEqual(center[1], 3.3)
# 
# 
#     def test_three_element(self):
#         points = np.array([[-np.sqrt(3)/2, 1/2], [-np.sqrt(3)/2, -1/2], [1, 0.0]])
#         center, radius = covers.minimum_covering_sphere_iterative(points)
#         self.assertAlmostEqual(radius, 1.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertAlmostEqual(center[0], 0.0)
#         self.assertAlmostEqual(center[1], 0.0)
# 
#     def test_many_element(self):
#         num_points = 500
#         points = np.random.normal(size=(num_points, 2))
#         points /= np.linalg.norm(points, axis=1).reshape(num_points, 1)
#         points[0] = np.array([-9, 0])
#         points[-1] = np.array([11, 0])
#         center, radius = covers.minimum_covering_sphere_iterative(points)
#         self.assertAlmostEqual(radius, 10.0)
#         self.assertIsInstance(center, np.ndarray)
#         self.assertEqual(len(center.shape), 1)
#         self.assertEqual(center.shape[0], 2)
#         self.assertAlmostEqual(center[0], 1)
#         self.assertAlmostEqual(center[1], 0)

class TestWelzlC(unittest.TestCase):
    def test_empty(self):
        points = np.empty((0,2))
        center, radius = covers.min_sphere(points)
        self.assertEqual(radius, 0.0)
        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(len(center.shape), 1)
        self.assertEqual(center.shape[0], 2)

    def test_one_element(self):
        points = np.array([[3.3,5.5]])
        center, radius = covers.min_sphere(points)
        self.assertEqual(radius, 0.0)
        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(len(center.shape), 1)
        self.assertEqual(center.shape[0], 2)
        self.assertEqual(center[0], 3.3)
        self.assertEqual(center[1], 5.5)

    def test_two_element(self):
        points = np.array([[0.2, 3.3], [2.2, 3.3]])
        center, radius = covers.min_sphere(points)
        self.assertAlmostEqual(radius, 1.0)
        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(len(center.shape), 1)
        self.assertEqual(center.shape[0], 2)
        self.assertAlmostEqual(center[0], 1.2)
        self.assertAlmostEqual(center[1], 3.3)

    def test_three_element(self):
        points = np.array([[-np.sqrt(3)/2, 1/2], [-np.sqrt(3)/2, -1/2], [1, 0.0]])
        center, radius = covers.min_sphere(points)
        self.assertAlmostEqual(radius, 1.0)
        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(len(center.shape), 1)
        self.assertEqual(center.shape[0], 2)
        self.assertAlmostEqual(center[0], 0.0)
        self.assertAlmostEqual(center[1], 0.0)

    def test_many_element(self):
        num_points = 500
        points = np.random.normal(size=(num_points, 2))
        points /= np.linalg.norm(points, axis=1).reshape(num_points, 1)
        points[0] = np.array([-9, 0])
        points[-1] = np.array([11, 0])
        center, radius = covers.min_sphere(points)
        self.assertAlmostEqual(radius, 10.0)
        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(len(center.shape), 1)
        self.assertEqual(center.shape[0], 2)
        self.assertAlmostEqual(center[0], 1)
        self.assertAlmostEqual(center[1], 0)


if __name__ == '__main__':
    unittest.main()

