import unittest
import numpy as np
from math import log, pi, sin
from astropy.units import s, rad
from pulsar import *

# constants of the project's initial profile
PHI = 1 * rad
D = 0.1
T = 0.01 * s # 10ms
IPEAK = 100

class TestSpeed(unittest.TestCase):

    def test_using_seconds(self):
        omega = speed(1 * s)
        self.assertEqual(omega.value, 2*pi)
        self.assertEqual(omega.unit, "rad / s")

    def test_using_ms(self):
        omega = speed(10**-2 * s)
        self.assertEqual(omega.value, 2*pi*100)
        self.assertEqual(omega.unit, "rad / s")

class TestConcentration(unittest.TestCase):

    def test_positive_duty_cycle(self):
        kappa = concentration(0.1)
        self.assertAlmostEqual(kappa, 14.1622, places=5)

    def test_zero_duty_cycle(self):
        with self.assertRaises(ZeroDivisionError):
            kappa = concentration(0)

class TestAngle(unittest.TestCase):

    def test_full_revolution(self):
        phi = angle(1 * rad, 10 * s, 10 * s)
        self.assertEqual(phi.value, 1 + 2*pi)

    def test_half_revolution(self):
        phi = angle(1 * rad, 10 * s, 5 * s)
        self.assertEqual(phi.value, 1 + pi)

class TestGaussianNoise(unittest.TestCase):

    def test_noise(self):
        np.random.seed(0)
        signal = np.array([0, 1, 2])
        noise = gaussian_noise(signal, .5)
        return self.assertIsNone(np.testing.assert_array_almost_equal(noise,
         [0.88, 0.2 , 0.49], decimal=2))

class TestBrightness(unittest.TestCase):

    """
    solve A*e^([log(2)/(2sin^2(D*pi/2))][cos((phi+((2pi)/T)*t)-1)])
    for A=100,phi=1rad,D=0.1,T=0.01s,t=0s
    """

    def test_brightness_at_t0(self):
        t = 0*s
        i = brightness(PHI, D, T, IPEAK, t)
        self.assertEqual(round(i.value, -3), 1.41437e8)

    def test_brightness_at_half_period(self):
        t = T/2
        i = brightness(PHI, D, T, IPEAK, t)
        self.assertAlmostEqual(i.value, 7.07028e-5, places=5)

    def test_brightness_at_1s(self):
        t = 1*s
        i = brightness(PHI, D, T, IPEAK, t)
        self.assertEqual(round(i.value, -3), 1.41437e8)

class TestLinearBrightness(unittest.TestCase):

    def test_linear_brightness(self):
        timeseries = np.linspace(0, 2, 1000) * s
        i = linear_brightness(PHI, D, T, IPEAK, timeseries)

        # highest
        self.assertEqual(round(i[0].value, -3), 1.41437e8)

        # lowest
        mid = timeseries.size//2
        self.assertEqual(round(i[mid].value, -3), 9.411e6)
        
        # highest
        self.assertEqual(round(i[-1].value, -3), 1.41437e8)

class TestSearchTemplates(unittest.TestCase):

    def test_at_peak_unit(self):
        timeseries = np.linspace(0, 2, 1000) * s
        params = [Parameters(.1, 2*pi*rad/(0.01*s), 1*rad)]
        templates = search_templates(timeseries, params)
        i = templates[0]

        # highest
        self.assertAlmostEqual(i[0].value, 1.414371e6, 0)

        # lowest
        mid = timeseries.size//2
        self.assertAlmostEqual(i[mid].value, 9.411e4, 0)
        
        # highest
        self.assertAlmostEqual(i[-1].value, 1.414371e6, 0)