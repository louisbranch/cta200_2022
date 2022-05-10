import numpy as np
from typing import Tuple

def mandelbrot_set(num_points = 50, max_iter = 10) -> np.array:
    """Implementation of a Madelbrot set using numpy.arrays.

    Parameters
    ----------

    num_points : int, optional
        The number of points for the x, y place
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

    # Create another 2D plane to store f(z) = z^2 + c and another to
    # store the number of iterations
    z = np.zeros((num_points, num_points), dtype=np.complex64)
    iter = np.zeros((num_points, num_points))

    for i in range(max_iter):
        z = z**2 + c
        mask = np.abs(z) > 2
        iter[mask] = i

    return iter