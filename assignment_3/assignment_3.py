import numpy as np
import warnings
from scipy.integrate import solve_ivp
from typing import List, Tuple

Sigma = 10.
Rayleigh = 28
Scale = 8./3.
Time = 60. # in seconds
DeltaT = 0.01

def mandelbrot_set(num_points = 50, max_iter = 10) -> np.array:
    """Implementation of a Madelbrot set using numpy.array.

    Parameters
    ----------

    num_points : int, optional
        The number of points for the x, y plane.
    max_iter : int, optional
        Maximum number of iterations to check whether the value is still
        bounded.

    Returns
    ------

    numpy.array
        2D array in which each point has the number of iterations performed
        to check for boundness.
    """

    # Create a complex plane c by adding two linear spaces as orthogonal
    # dimensions: c = x + yi
    x = np.linspace(-2, 2, num=num_points, dtype=np.float64)
    y = np.linspace(-2, 2, num=num_points, dtype=np.complex64) * 1j
    c = x.reshape((1,num_points)) + y.reshape((num_points,1))

    # Create another 2D array to store f(z) = z^2 + c and another to
    # store the number of iterations
    z = np.zeros((num_points, num_points), dtype=np.complex64)
    iter = np.zeros((num_points, num_points))

    for i in range(max_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = z**2 + c
            mask = np.abs(z) > 2
        iter[mask] = i

    return iter

def lorenz_integral(space : np.array, w0 = [0., 1., 0.]) -> Tuple[List, List]:
    """Solve a system of ordinary equations simulating chaotic atmospheric
    conditions.

    Parameters
    ----------

    space : numpy.array
        Set of points for the solution.
    w0 : List, optional
        Initial conditions for the system.

    Returns
    ------

    Tuple[List, List]
        Tuple containing the timestamps and solutions for X, Y, Z.
    """

    def lorenz(t: float, xyz: Tuple[float, float, float], sigma: float,
     r: float, b: float) -> Tuple[float, float, float]:

        x, y, z = xyz

        dx = -sigma * (x - y)
        dy = r*x - y - x*z
        dz = -b*z + x*y

        return dx, dy, dz

    tspan = (0., Time)
    args = (Sigma, Rayleigh, Scale)

    result = solve_ivp(lorenz, tspan, w0, args=args, dense_output=True)

    sol = result.sol(space)

    return space / DeltaT, sol
