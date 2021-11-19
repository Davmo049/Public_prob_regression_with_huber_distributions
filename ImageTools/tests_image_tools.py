import unittest

import numpy as np

from . import ImageTools

class TestAffineStack(unittest.TestCase):
    def test_translation(self):
        pos = np.array([1.0,2])
        trans = np.array([7.0,3])
        transform = ImageTools.translation_as_affine(trans)
        new_pos = transform(pos)

        self.assertAlmostEqual(new_pos[0], 8)
        self.assertAlmostEqual(new_pos[1], 5)

    def test_rotation(self):
        pos = np.array([2.0,3.0])
        center = np.array([1.0,2])
        angle = np.pi/4
        transform = ImageTools.rotate_as_affine(angle, center)
        new_pos = transform(pos)
        self.assertAlmostEqual(new_pos[0], 1)
        self.assertAlmostEqual(new_pos[1], 2+np.sqrt(2))

    def test_scale(self):
        pos = np.array([1.0,1.0])
        center = np.array([0.0,0])
        angle = np.pi/4
        scale=np.array([10, 3])
        transform = ImageTools.scale_as_affine(angle, scale, center)
        new_pos = transform(pos)
        self.assertAlmostEqual(new_pos[0], 10)
        self.assertAlmostEqual(new_pos[1], 10)

    def test_perspective_as_affine(self):
        pos = np.array([1.0,1.0])
        center = np.array([0.0,0])
        angle = np.pi/4
        strength = -0.5/np.sqrt(2)
        transform = ImageTools.perspective_as_affine(angle, strength)
        new_pos = transform(pos)
        self.assertAlmostEqual(new_pos[0], 2)
        self.assertAlmostEqual(new_pos[1], 2)

