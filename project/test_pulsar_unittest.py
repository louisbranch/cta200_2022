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

    peak = 100
    phi = 1 * u.rad
    d = 0.1
    period = 0.01 * u.s # 10ms

    def test_brightness_at_t0(self):
        t = 0*u.s
        i = brightness(self.phi, self.d, self.period, self.peak, t)
        self.assertEqual(round(i.value, -3), 1.41437e8)

    def test_brightness_at_half_period(self):
        t = self.period/2
        i = brightness(self.phi, self.d, self.period, self.peak, t)
        self.assertAlmostEqual(i.value, 7.07028e-5, places=5)

    def test_brightness_at_1s(self):
        t = 1*u.s
        i = brightness(self.phi, self.d, self.period, self.peak, t)
        self.assertEqual(round(i.value, -3), 1.41437e8)

class TestLinearBrightness(unittest.TestCase):

    peak = 100
    phi = 1 * u.rad
    d = 0.1
    period = 0.01 * u.s # 10ms

    def test_linear_brightness(self):
        tframe = np.linspace(0, 2, 1000) * u.s
        i = linear_brightness(self.phi, self.d, self.period, self.peak, tframe)

        # highest
        self.assertEqual(round(i[0].value, -3), 1.41437e8)

        # lowest
        mid = tframe.size//2
        self.assertEqual(round(i[mid].value, -3), 9.411e6)
        
        # highest
        self.assertEqual(round(i[-1].value, -3), 1.41437e8)