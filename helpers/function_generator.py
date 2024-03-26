import numpy as np
import math


def get_function_samples(samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset which follows the function f(x) = cos(x) + sin(2x)

    Parameters
    ----------
    samples : int
        Amount of samples to generate
        Default value is 2000 samples

    Returns
    -------
    x, y : tuple[np.ndarray, np.ndarray]
        x- and y- coordinates (features and labels)
    """

    # Generate random x coordinates between 0 and 2pi, shuffle their order
    x = np.random.uniform(0, 2 * math.pi, size=samples)
    np.random.shuffle(x)

    # Generate y values with noise
    y = np.cos(x) + np.sin(2 * x)
    y += 0.1 * np.random.randn(*y.shape)

    return x, y

