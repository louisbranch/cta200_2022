import unittest

from assignment_2 import f, df, newton

class TestF(unittest.TestCase):

    def test_positive_number(self):
        n = f(2)
        self.assertEqual(n, 3)
    
    def test_negative_number(self):
        n = f(-2)
        self.assertEqual(n, -13)

class TestDF(unittest.TestCase):

    def test_positive_number(self):
        n = df(2)
        self.assertEqual(n, 8)
    
    def test_negative_number(self):
        n = df(-2)
        self.assertEqual(n, 16)

class TestNewton(unittest.TestCase):

    def test_converging_result(self):
        n = newton(f, df, 10)
        self.assertAlmostEqual(n, 1.465571231876768)

    def test_diverging_result(self):
        nf = lambda x: x**2 + 1
        ndf = lambda x: 2*x
        n = newton(nf, ndf, 10)
        self.assertIsNone(n)

    def test_zero_derivative(self):
        ndf = lambda x: 0
        n = newton(f, ndf, 10)
        self.assertIsNone(n)