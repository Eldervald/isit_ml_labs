import numpy as np
import numpy.typing as npt


def vander(array: npt.NDArray[np.float64 | np.int64]) -> npt.NDArray[np.float64]:
    """
    Create a Vandermod matrix from the given vector.
    :param array: input array,
    :return: vandermonde matrix
    """
