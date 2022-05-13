import unittest
import numpy as np
from math import log, pi, sin
from astropy import units as u
from pulsar import *

class TestSpeed(unittest.TestCase):

    def test_using_seconds(self):
        omega = speed(1 * u.s)
        self.assertEqual(omega.value, 2*pi)
        self.assertEqual(omega.unit, u.Unit("rad / s"))

    def test_using_ms(self):
        omega = speed(10**-2 * u.s)
        self.assertEqual(omega.value, 2*pi*100)
        self.assertEqual(omega.unit, u.Unit("rad / s"))

class TestConcentration(unittest.TestCase):

    def test_positive_duty_cycle(self):
        kappa = concentration(0.1)
        self.assertAlmostEqual(kappa, 14.1622, places=5)

    def test_zero_duty_cycle(self):
        with self.assertRaises(ZeroDivisionError):
            kappa = concentration(0)

class TestAngle(unittest.TestCase):

    def test_full_revolution(self):
        phi = angle(1 * u.rad, 10 * u.s, 10 * u.s)
        self.assertEqual(phi.value, 1 + 2*pi)

    def test_half_revolution(self):
        phi = angle(1 * u.rad, 10 * u.s, 5 * u.s)
        self.assertEqual(phi.value, 1 + pi)

class TestGaussianNoise(unittest.TestCase):

    def test_small_noise(self):
        np.random.seed(1)
        signal = np.array([0, 1])
        noise = gaussian_noise(signal, 0.2)
        self.assertIsNone(np.testing.assert_almost_equal(noise,
         [0.32, 0.87], decimal=2))

class TestIntensity(unittest.TestCase):

    """
    solve A*e^([log(2)/(2sin^2(D*pi/2))][cos((phi+((2pi)/T)*t)-1)])
    for A=100,phi=1rad,D=0.1,T=0.01s,t=0s
    """

    peak = 100
    phi = 1 * u.rad
    d = 0.1
    period = 0.01 * u.s # 10ms

    def test_intensity_at_t0(self):
        val = intensity(self.peak, self.phi, self.d, self.period, 0*u.s)
        self.assertEqual(round(val, -3), 1.41437e8)

    def test_intensity_at_half_period(self):
        val = intensity(self.peak, self.phi, self.d, self.period, self.period/2)
        self.assertAlmostEqual(val, 7.07028e-5, places=5)