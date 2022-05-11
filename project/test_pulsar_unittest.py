import unittest
from math import log, pi, sin
from astropy import units as u
from pulsar import speed, concentration, angle

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
        self.assertAlmostEqual(kappa, 0.00848126, places=8)

    def test_zero_duty_cycle(self):
        kappa = concentration(0)
        self.assertEqual(kappa, 0)

class TestAngle(unittest.TestCase):

    def test_full_revolution(self):
        phi = angle(1 * u.rad, 10 * u.s, 10 * u.s)
        self.assertEqual(phi.value, 1 + 2*pi)

    def test_half_revolution(self):
        phi = angle(1 * u.rad, 10 * u.s, 5 * u.s)
        self.assertEqual(phi.value, 1 + pi)