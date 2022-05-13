from math import log, sin, pi
import numpy as np
from astropy import units as u
from scipy.integrate import quad
from typing import Any

MU = 1 * u.rad

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

def brightness(phi0: u.rad, d: float, period: u.s, peak: float,
 time: u.s) -> float:

    kappa = concentration(d)
    phi = angle(phi0, period, time)
    return peak * np.exp(kappa * np.cos(phi - MU))

def linear_brightness(phi0: u.rad, d: float, period: u.s, peak: float,
timeline: np.array) -> np.array:

    mapping = lambda t: brightness(phi0, d, period, peak, t)

    return np.apply_along_axis(mapping, 0, timeline)

def noisy_brightness(phi0: u.rad, d: float, period: u.s, peak: float,
stddev: float, timeline: np.array) -> np.array:

    noise = gaussian_noise(timeline, stddev)
    acc = []

    for n, time in zip(noise, timeline):
        b = brightness(phi0, d, period, peak+n, time)
        acc.append(b)

    return acc

def integrated_brightness(phi0: u.rad, d: float, period: u.s,
ts: u.s, peak: float, steps: np.array):

    integrand = lambda t : brightness(phi0, d, period, peak, t*u.s)

    k_integrated = []

    for i in range(len(steps)-1):
        result, err = quad(integrand, steps[i], steps[i+1])
        k_integrated.append(result / ts.value)

    return k_integrated

def gaussian_noise(signal: np.array, stddev: float):
    n = signal.shape
    return np.random.normal(0, stddev, n)