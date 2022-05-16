from math import log, sin, pi
import numpy as np
from astropy.units import s, rad
from scipy.integrate import quad
from collections import namedtuple
from typing import List

# Measure of location (the distribution is clustered around mu).
# Analogous to the mean in a normal distribuition.
MU = 1 * rad

# Search parameters used for linear regression
Parameters = namedtuple('Parameters', ['D', 'omega', 'phi0'])

def concentration(d: float) -> float:
    """Concentration parameter (reciprocal of dispersion) for d, the duty
    cycle value.

    Parameters
    ----------
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.

    Returns
    -------

    float
        Concentration parameter, analogous to the inverse of variance in a
        normal distribuition.
    """
    return log(2) / (2 * (sin(pi * d / 2) ** 2))

def speed(period: s) -> rad/s:
    """Angular speed for a given period.

    Parameters
    ----------

    period: s
        The time for one revolution around the circle in seconds.

    Returns
    -------

    rad/s
        Angular speed or frequency in radians per second.
    """
    return (2 * pi * rad) / period

def angle(phi0: rad, period: s, dt: s) -> rad:
    """Angular displacement given an initial angle and angular speed.

    Parameters:
    -----------

    phi0: rad
        Initial angle value in radians at t0.
    period: s
        The time in seconds for one revolution around the circle.
    dt: s
        Delta time in seconds between t - t0.

    Returns
    -------

    rad
        Total angular displacement in radians after the delta time has ellapsed.
    """
    return phi0 + speed(period) * dt

def brightness(phi0: rad, d: float, period: s, peak: float,
 time: s) -> float:
    """
    Observed brightness of a pulsar using a von Mises distribution for a given
    time.

    Parameters:
    -----------

    phi0: rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    time: s
        Ellapsed time in seconds.

    Returns
    -------

    float
        Observed brightness at a given time.
    """

    kappa = concentration(d)
    phi = angle(phi0, period, time)
    return peak * np.exp(kappa * np.cos(phi - MU))

def linear_brightness(phi0: rad, d: float, period: s, peak: float,
timeseries: np.array) -> np.array:
    """
    Observed brightness of a pulsar using a von Mises distribution for a time
    series.

    Parameters:
    -----------

    phi0: rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    timeseries: np.array
        Time series in which the first dimension contains time values in 
        seconds.

    Returns
    -------

    np.array
        Observed brightness at each data point in the time series.
    """

    mapping = lambda t: brightness(phi0, d, period, peak, t)

    return np.apply_along_axis(mapping, 0, timeseries)

def noisy_brightness(phi0: rad, d: float, period: s, peak: float,
stddev: float, timeseries: np.array) -> np.array:
    """
    Observed brightness of a pulsar using a von Mises distribution for a time
    series with gaussian noise for a given standard deviation.

    Parameters:
    -----------

    phi0: rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    stddev: float
        Amount of dispersion for the peak values.
    timeseries: np.array
        Time series in which the first dimension contains time values in 
        seconds.

    Returns
    -------

    np.array
        Observed brightness with noise at each data point in the time series.
    """

    noise = gaussian_noise(timeseries, stddev)
    acc = []

    for n, time in zip(noise, timeseries):
        b = brightness(phi0, d, period, peak+n, time)
        acc.append(b)

    return acc

def integrated_brightness(phi0: rad, d: float, period: s,
peak: float, dt: s, steps: np.array):
    """
    Observed brightness of a pulsar using a von Mises distribution integrated
    over time.

    Parameters:
    -----------

    phi0: rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    dt: s
        Delta time between each integration step.
    steps: np.array
        Series of steps to integrate from.

    Returns
    -------

    np.array
        Observed brightness at each data point in the time series.
    """

    integrand = lambda t : brightness(phi0, d, period, peak, t*s)

    acc = []

    for i in range(len(steps)-1):
        result, err = quad(integrand, steps[i], steps[i+1])
        acc.append(result / dt.value)

    return np.array(acc)

def gaussian_noise(signal: np.array, stddev: float) -> np.array:
    """
    Signal noise that has a probability density function equal to the normal
    distribution.

    Parameters
    ----------

    signal: np.array
        Array to be used as the shape of the noise.
    stddev: float
        Amount of dispersion for the signal values.

    Returns
    -------

    np.array
        Array with the same shape as the original containing the random noise.
    """
    n = signal.shape
    return np.random.normal(0, stddev, n)

def search_parameters() -> List:
    Parameters = namedtuple('Parameters', ['D', 'omega', 'phi0'])

    omega = lambda T: (2*pi*rad)/(T*s)

    return [
        Parameters(.2,  omega(.025), 0*rad),
        Parameters(.4,  omega(.005), .25*rad),
        Parameters(.8,  omega(.075), .5*rad),
        Parameters(.1,  omega(.01),  1*rad), # original values
        Parameters(.12, omega(.02), .75*rad),
        Parameters(.16, omega(.05), .95*rad),
    ]

def search_templates(dt: s, steps: np.array, params=[]) -> np.array:
    peak = 1 # unit

    if len(params) == 0:
        params = search_parameters()

    acc = []
    for d, omega, phi0 in params:
        ik = []
        period = (2*pi*rad)/omega
        integrand = lambda t : brightness(phi0, d, period, peak, t*s)
        for i in range(len(steps)-1):
            result, err = quad(integrand, steps[i], steps[i+1])
            ik.append(result / dt.value)
        acc.append(ik)
    return np.array(acc)

def measurement(timeseries: np.array, params=[]) -> np.array:
    return None

def brightness_estimator(tk: np.array, dk: np.array) -> float:
    return np.sum(np.dot(tk, dk)) / np.sum(np.dot(tk, tk))