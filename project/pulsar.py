from math import log, sin, pi
import numpy as np
from astropy import units as u
from scipy.integrate import solve_ivp

def concentration(d: float) -> float:
    """ Calculate concentration (reciprocal of dispersion) using d, the duty
    cycle value.
    """
    return log(2) / 2 * sin(pi * d / 2) ** 2

def speed(period: u.s) -> u.rad/u.s:
    return (2 * pi * u.rad) / period

def angle(phi0: u.rad, period: u.s, time: u.s) -> u.rad:
    return phi0 + speed(period) * time

def intensity(peak: float, phi0: u.rad, d: float, period: u.s, 
tframe: np.array) -> np.array:

    kappa = concentration(d)
    
    # For each time step, calculate the angular speed using the period.
    # Use the angle at time t to calculate the intensity based on the 
    # concentration.
    def mapping(time: u.s):
        phi = angle(phi0, period, time)
        return peak * np.exp(kappa * np.cos(phi.value - 1))

    return np.apply_along_axis(mapping, 0, tframe)

def integrated_intensity(peak: float, phi0: u.rad, d:float, period: u.s, 
interval: u.s, k: float):

    y0 = (0,)
    tspan = (k, k + 1)
    args = (1,)

    def integration(time: u.s):
        return None

    return solve_ivp(integration, tspan, y0, args=args, dense_output=True)