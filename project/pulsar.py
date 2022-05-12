from math import log, sin, pi
import numpy as np
from astropy import units as u
from scipy.integrate import quad
from typing import Any

def concentration(d: float) -> float:
    """Calculate concentration (reciprocal of dispersion) using d, the duty
    cycle value.
    """
    return log(2) / (2 * (sin(pi * d / 2) ** 2))

def speed(period: u.s) -> u.rad/u.s:
    """Calculate the angular speed for a given period."""
    return (2 * pi * u.rad) / period

def angle(phi0: u.rad, period: u.s, dt: u.s) -> u.rad:
    """Calculate the angular displacement given an initial value and angular
    speed.
    """
    return phi0 + speed(period) * dt

def intensity(peak: float, phi0: u.rad, d: float, period: u.s,
 time: u.s) -> float:

    kappa = concentration(d)
    phi = angle(phi0, period, time)
    return peak * np.exp(kappa * np.cos(phi.value - 1))

def linear_intensity(peak: float, phi0: u.rad, d: float, period: u.s, 
tframe: np.array) -> np.array:

    mapping = lambda t: intensity(peak, phi0, d, period, t)

    return np.apply_along_axis(mapping, 0, tframe)

def integrated_intensity(peak: float, phi0: u.rad, d: float, period: u.s,
ts: u.s, steps: np.array):

    integrand = lambda t : intensity(peak, phi0, d, period, t*ts)

    k_integrated = []

    for k in steps:
        result, err = quad(integrand, k, k+1)
        k_integrated.append(result / ts.value)

    return k_integrated