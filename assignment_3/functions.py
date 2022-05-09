import numpy as np
from typing import Tuple

def mandelbrot_set(num_points = 50, max_iter = 10) -> np.array:

    x = np.linspace(-2, 2, num=num_points, dtype=np.float64)
    y = np.linspace(-2, 2, num=num_points, dtype=np.complex64) * 1j
    c = x.reshape((1,num_points)) + y.reshape((num_points,1))

    z = np.zeros((num_points, num_points), dtype=np.complex128)

    for i in range(max_iter):
        z = z**2 + c

    return z
