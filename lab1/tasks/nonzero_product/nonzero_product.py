import numpy as np
import numpy.typing as npt


def nonzero_product(matrix: npt.NDArray[np.int64]) -> int | None:
    """
    Compute product of nonzero diagonal elements of matrix
    If all diagonal elements are zeros, then return None
    :param matrix: array,
    :return: product value or None
    """
