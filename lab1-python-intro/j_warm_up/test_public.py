from dataclasses import dataclass
from types import GeneratorType
from typing import Any

import pytest
import testlib

from .warm_up import transpose, uniq, dict_merge, product


###################
# Structure asserts
###################


def test_docs() -> None:
    assert testlib.is_function_docstring_exists(transpose)
    assert testlib.is_function_docstring_exists(uniq)
    assert testlib.is_function_docstring_exists(dict_merge)
    assert testlib.is_function_docstring_exists(product)


###################
# Tests
###################


@dataclass
class TransposeTestCase:
    name: str
    matrix: list[list[Any]]
    result: list[list[Any]]

    def __str__(self) -> str:
        return self.name


TRANSPOSE_TEST_CASES = [
    TransposeTestCase(
        name='rectangular',
        matrix=[[1, 2], [3, 4], [5, 6]],
        result=[[1, 3, 5], [2, 4, 6]],
    ),
    TransposeTestCase(
        name='square',
        matrix=[[1, 2], [3, 4]],
        result=[[1, 3], [2, 4]],
    ),
    TransposeTestCase(
        name='single_element_rows',
        matrix=[[1], [3]],
        result=[[1, 3]],
    ),
    TransposeTestCase(
        name='single_row',
        matrix=[[1, 3]],
        result=[[1], [3]],
    ),
    TransposeTestCase(
        name='single_element',
        matrix=[[1]],
        result=[[1]],
    )
]


@pytest.mark.parametrize('test_case', TRANSPOSE_TEST_CASES, ids=str)
def test_transpose(test_case: TransposeTestCase) -> None:
    assert transpose(test_case.matrix) == test_case.result


@dataclass
class UniqTestCase:
    name: str
    sequence: list[Any]
    expected: list[Any]

    def __str__(self) -> str:
        return self.name


UNIQ_TEST_CASES = [
    UniqTestCase(
        name='complex',
        sequence=[1, 2, 3, 3, 1, 7],
        expected=[1, 2, 3, 7],
    ),
    UniqTestCase(
        name='two_elements',
        sequence=[1, 1, 3, 1, 1, 3],
        expected=[1, 3],
    ),
    UniqTestCase(
        name='single_element',
        sequence=[1],
        expected=[1],
    ),
    UniqTestCase(
        name='empty',
        sequence=[],
        expected=[],
    )
]


@pytest.mark.parametrize('test_case', UNIQ_TEST_CASES, ids=str)
def test_uniq(test_case: UniqTestCase) -> None:
    generator = uniq(test_case.sequence)
    assert isinstance(generator, GeneratorType)
    assert list(generator) == test_case.expected


@dataclass
class DictMergeTestCase:
    name: str
    dicts: list[dict[Any, Any]]
    expected: dict[Any, Any]

    def __str__(self) -> str:
        return self.name


DICT_MERGE_TEST_CASES = [
    DictMergeTestCase(
        name='complex',
        dicts=[{1: 2}, {2: 2}, {1: 1}],
        expected={1: 1, 2: 2},
    ),
    DictMergeTestCase(
        name='common_key',
        dicts=[{1: 2}, {1: 3}, {1: 1}],
        expected={1: 1},
    ),
    DictMergeTestCase(
        name='different_keys',
        dicts=[{1: 2}, {2: 3}, {3: 4}],
        expected={1: 2, 2: 3, 3: 4},
    ),
    DictMergeTestCase(
        name='empty_dicts',
        dicts=[{}, {}, {3: 4}],
        expected={3: 4},
    )
]


@pytest.mark.parametrize('test_case', DICT_MERGE_TEST_CASES, ids=str)
def test_dict_merge(test_case: DictMergeTestCase) -> None:
    assert dict_merge(*test_case.dicts) == test_case.expected


@dataclass
class ProductTestCase:
    name: str
    left: list[int]
    right: list[int]
    expected: int

    def __str__(self) -> str:
        return self.name


PRODUCT_TEST_CASES = [
    ProductTestCase(
        name='three_element',
        left=[1, 2, 3],
        right=[4, 5, 6],
        expected=32,
    ),
    ProductTestCase(
        name='two_element',
        left=[1, 2],
        right=[3, 4],
        expected=11,
    ),
    ProductTestCase(
        name='single_element',
        left=[1],
        right=[1],
        expected=1,
    )
]


@pytest.mark.parametrize('test_case', PRODUCT_TEST_CASES, ids=str)
def test_product(test_case: ProductTestCase) -> None:
    assert product(test_case.left, test_case.right) == test_case.expected


###################
# Additional edge case tests
###################


def test_transpose_empty() -> None:
    # Edge case: empty matrix
    assert transpose([]) == []


def test_transpose_input_immutability() -> None:
    # Edge case: verify input is not modified
    original = [[1, 2], [3, 4]]
    original_copy = [row[:] for row in original]
    transpose(original)
    assert original == original_copy


def test_uniq_generator_input() -> None:
    # Edge case: generator input validation
    def gen():
        yield 1
        yield 2
        yield 2
        yield 3

    result = list(uniq(gen()))
    assert result == [1, 2, 3]


def test_dict_merge_no_args() -> None:
    # Edge case: no arguments
    assert dict_merge() == {}


def test_dict_merge_immutability() -> None:
    # Edge case: verify inputs are not modified
    d1 = {1: 2}
    d2 = {3: 4}
    d1_copy = d1.copy()
    d2_copy = d2.copy()
    dict_merge(d1, d2)
    assert d1 == d1_copy
    assert d2 == d2_copy


def test_product_empty_vectors() -> None:
    # Edge case: empty vectors
    assert product([], []) == 0


def test_product_input_immutability() -> None:
    # Edge case: verify inputs are not modified
    left = [1, 2, 3]
    right = [4, 5, 6]
    left_copy = left[:]
    right_copy = right[:]
    product(left, right)
    assert left == left_copy
    assert right == right_copy
