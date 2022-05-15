from math import log, sin, pi
import numpy as np
from astropy import units as u
from scipy.integrate import quad

# Measure of location (the distribution is clustered around mu).
# Analogous to the mean in a normal distribuition.
MU = 1 * u.rad

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

def speed(period: u.s) -> u.rad/u.s:
    """Angular speed for a given period.

    Parameters
    ----------

    period: u.s
        The time for one revolution around the circle in seconds.

    Returns
    -------

    u.rad/u.s
        Angular speed or frequency in radians per second.
    """
    return (2 * pi * u.rad) / period

def angle(phi0: u.rad, period: u.s, dt: u.s) -> u.rad:
    """Angular displacement given an initial angle and angular speed.

    Parameters:
    -----------

    phi0: u.rad
        Initial angle value in radians at t0.
    period: u.s
        The time in seconds for one revolution around the circle.
    dt: u.s
        Delta time in seconds between t - t0.

    Returns
    -------

    u.rad
        Total angular displacement in radians after the delta time has ellapsed.
    """
    return phi0 + speed(period) * dt

def brightness(phi0: u.rad, d: float, period: u.s, peak: float,
 time: u.s) -> float:
    """
    Observed brightness of a pulsar using a von Mises distribution for a given
    time.

    Parameters:
    -----------

    phi0: u.rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: u.s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    time: u.s
        Ellapsed time in seconds.

    Returns
    -------

    float
        Observed brightness at a given time.
    """

    kappa = concentration(d)
    phi = angle(phi0, period, time)
    return peak * np.exp(kappa * np.cos(phi - MU))

def linear_brightness(phi0: u.rad, d: float, period: u.s, peak: float,
timeline: np.array) -> np.array:
    """
    Observed brightness of a pulsar using a von Mises distribution for a time
    series.

    Parameters:
    -----------

    phi0: u.rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: u.s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    timelime: np.array
        Time series in which the first dimension contains time values in 
        seconds.

    Returns
    -------

    np.array
        Observed brightness at each data point in the time series.
    """

    mapping = lambda t: brightness(phi0, d, period, peak, t)

    return np.apply_along_axis(mapping, 0, timeline)

def noisy_brightness(phi0: u.rad, d: float, period: u.s, peak: float,
stddev: float, timeline: np.array) -> np.array:
    """
    Observed brightness of a pulsar using a von Mises distribution for a time
    series with gaussian noise for a given standard deviation.

    Parameters:
    -----------

    phi0: u.rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: u.s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    stddev: float
        Amount of dispersion for the peak values.
    timelime: np.array
        Time series in which the first dimension contains time values in 
        seconds.

    Returns
    -------

    np.array
        Observed brightness with noise at each data point in the time series.
    """

    noise = gaussian_noise(timeline, stddev)
    acc = []

    for n, time in zip(noise, timeline):
        b = brightness(phi0, d, period, peak+n, time)
        acc.append(b)

    return acc

def integrated_brightness(phi0: u.rad, d: float, period: u.s,
peak: float, dt: u.s, steps: np.array):
    """
    Observed brightness of a pulsar using a von Mises distribution integrated
    over time.

    Parameters:
    -----------

    phi0: u.rad
        Initial angle value in radians at t0.
    d : float
        Duty cycle of the pulsar, usually between 0 and 0.16.
    period: u.s
        The time in seconds for one revolution around the circle.
    peak: float
        Peak brightness of the pulsar.
    dt: u.s
        Delta time between each integration step.
    steps: np.array
        Series of steps to integrate from.

    Returns
    -------

    np.array
        Observed brightness at each data point in the time series.
    """

    integrand = lambda t : brightness(phi0, d, period, peak, t*u.s)

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

def search_template(phi0: u.rad, d: float, period: u.s,
 timeline: np.array) -> np.array:
    peak = 1
    return linear_brightness(phi0, d, period, peak, timeline)