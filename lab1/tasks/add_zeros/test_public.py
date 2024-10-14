import dataclasses

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

from .add_zeros import add_zeros


@dataclasses.dataclass
class AddZerosCase:
    array: npt.NDArray[np.int64]
    result: npt.NDArray[np.int64]


ADD_ZEROS_TEST_CASES = [
    AddZerosCase(
        array=np.array([1, 2, 3]),
        result=np.array([1, 0, 2, 0, 3])),
    AddZerosCase(
        array=np.array([], dtype=np.int64),
        result=np.array([], dtype=np.int64)),
    AddZerosCase(
        array=np.array([1]),
        result=np.array([1])),
    AddZerosCase(
        array=np.array([1, 1]),
        result=np.array([1, 0, 1])),
    AddZerosCase(
        array=np.array([0]),
        result=np.array([0])),
    AddZerosCase(
        array=np.array([1, 0, 0, 1]),
        result=np.array([1, 0, 0, 0, 0, 0, 1])),
    AddZerosCase(
        array=np.zeros(10, dtype=np.int64),
        result=np.zeros(19, dtype=np.int64)),
]


@pytest.mark.parametrize('t', ADD_ZEROS_TEST_CASES, ids=str)
def test_construct_matrix(t: AddZerosCase) -> None:
    assert_array_equal(add_zeros(t.array), t.result)
