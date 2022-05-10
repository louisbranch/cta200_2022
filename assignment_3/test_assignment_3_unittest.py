import unittest
import numpy as np

from assignment_3 import mandelbrot_set

class TestMandelbrotSet(unittest.TestCase):

    def test_few_points(self):
        set, mask = mandelbrot_set(5, max_iter = 10)
        self.assertEqual(set.shape, (5, 5))
        self.assertEqual(mask.shape, (5, 5))
