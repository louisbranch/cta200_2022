from typing import Callable, Optional

def f(x: float) -> float:
    """Calculates: x^3 - x^2 - 1

    Parameters
    ----------
    x : float

    Returns
    -------
    float
        Result of the equation.
    """
    return x**3 - x**2 - 1

def df(x: float) -> float:
    """First derivative of: x^3 - x^2 - 1

    Parameters
    ----------
    x : float

    Returns
    -------
    float
        Result of the first derivative.
    """
    return 3*x**2 - 2*x

def newton(f: Callable, df: Callable, x0: float, epsilon=1e-6,
 max_iter=30) -> Optional[float]:
    """Root-finding algorithm using Newton's method.

    Parameters
    ----------
    f : real function
        A function that accepts and returns a float.
    df : real function
        First derivative of `f`.
    x0 : float
        Initial guess for the root of `f`.
    epsilon : float, optional
        Result has to be smaller than this threshold value. 
    max_iter : int, optional
        Maximum number of iterations for the approximation.

    Returns
    -------
    float or None
        Root of the function if the iteration is successful, None otherwise.
    """
    
    xn = x0

    for i in range(max_iter):
        fxn = f(xn)

        if abs(fxn) < epsilon:
            print(f'Found root in {i} iterations')
            return xn

        dfxn = df(xn)
        if dfxn == 0:
            break

        xn -= fxn/dfxn

    print('Iteration failed')
    return None