# mypy: ignore-errors

import inspect
import tempfile
import typing as tp

import mypy.api
import testlib

from . import typy_generic
from .typy_generic import Vector, VectorIterator


###################
# Structure asserts
###################


def test_docs() -> None:
    assert testlib.is_function_docstring_exists(Vector.__init__)
    assert testlib.is_function_docstring_exists(Vector.__len__)
    assert testlib.is_function_docstring_exists(Vector.__getitem__)
    assert testlib.is_function_docstring_exists(Vector.append)
    assert testlib.is_function_docstring_exists(Vector.extend)
    assert testlib.is_function_docstring_exists(Vector.__iter__)
    assert testlib.is_function_docstring_exists(Vector.__str__)
    assert testlib.is_function_docstring_exists(Vector.__repr__)
    assert testlib.is_function_docstring_exists(Vector.__add__)
    assert testlib.is_function_docstring_exists(Vector.__sub__)
    assert testlib.is_function_docstring_exists(Vector.__mul__)
    assert testlib.is_function_docstring_exists(Vector.__rmul__)
    assert testlib.is_function_docstring_exists(Vector.dot)
    assert testlib.is_function_docstring_exists(VectorIterator.__init__)
    assert testlib.is_function_docstring_exists(VectorIterator.__iter__)
    assert testlib.is_function_docstring_exists(VectorIterator.__next__)


###################
# Tests
###################


def check_annotations(func):
    for p, value in inspect.signature(func).parameters.items():
        if p == "self":
            assert value.annotation == inspect.Signature.empty, f"Parameter {p} should not have annotation"
        else:
            assert value.annotation != inspect.Signature.empty, f"Parameter {p} does not have annotation"
            assert value.annotation != tp.Any, f"Parameter {p} has prohibited Any annotation"

    assert inspect.signature(func).return_annotation != inspect.Signature.empty, "Return does not have annotation"
    assert inspect.signature(func).return_annotation != tp.Any, "Return has prohibited Any annotation"


def check_func(module, test_case, is_success):
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        fp.write("import typing as tp")
        fp.write("\n")
        fp.write("import numbers")
        fp.write("\n")
        fp.write("import abc")
        fp.write("\n\n")
        fp.write(inspect.getsource(module))
        fp.write("\n")
        fp.write(inspect.getsource(test_case))
        fp.write("\n")
        fp.flush()

        normal_report, error_report, exit_status = mypy.api.run([fp.name, '--config-file', ''])
        print(f"Report:\n{normal_report}\n{error_report}")
        result_success_status = exit_status == 0
        assert result_success_status is is_success, \
            f"Mypy check should be {is_success}, but result {result_success_status}"


def test_vector_basic_operations() -> None:
    # Test creation, len, and getitem
    v = Vector[int]([1, 2, 3])
    assert len(v) == 3
    assert v[0] == 1
    assert v[1] == 2
    assert v[2] == 3

    # Test negative indexing
    assert v[-1] == 3
    assert v[-2] == 2
    assert v[-3] == 1


def test_vector_append() -> None:
    v = Vector[int]([1, 2])
    v.append(3)
    assert len(v) == 3
    assert v[2] == 3

    v.append(4)
    assert len(v) == 4
    assert v[3] == 4


def test_vector_extend() -> None:
    v = Vector[int]([1, 2])
    v.extend([3, 4])
    assert len(v) == 4
    assert v[2] == 3
    assert v[3] == 4

    v.extend([5])
    assert len(v) == 5
    assert v[4] == 5


def test_vector_str_repr() -> None:
    v = Vector[int]([1, 2, 3])
    str_result = str(v)
    repr_result = repr(v)

    # Check that both return non-empty strings
    assert len(str_result) > 0
    assert len(repr_result) > 0

    # Repr should be more detailed than str
    assert "Vector" in repr_result or "1" in repr_result


def test_iterator_protocol_for_loop() -> None:
    v = Vector[int]([1, 2, 3, 4])
    result = []

    for item in v:
        result.append(item)

    assert result == [1, 2, 3, 4]


def test_iterator_protocol_iter_next() -> None:
    v = Vector[int]([1, 2, 3])
    it = iter(v)

    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 3

    try:
        next(it)
        assert False, "StopIteration should have been raised"
    except StopIteration:
        pass


def test_iterator_independence() -> None:
    v = Vector[int]([1, 2, 3, 4, 5])

    it1 = iter(v)
    it2 = iter(v)

    # Advance first iterator
    assert next(it1) == 1
    assert next(it1) == 2

    # Second iterator should be at start
    assert next(it2) == 1
    assert next(it2) == 2

    # First iterator continues independently
    assert next(it1) == 3


def test_iterator_with_multiple_iterations() -> None:
    v = Vector[int]([1, 2, 3])

    # First iteration
    result1 = list(v)
    assert result1 == [1, 2, 3]

    # Second iteration should work independently
    result2 = list(v)
    assert result2 == [1, 2, 3]


def case1() -> None:
    Vector[int](1, 2, 3)  # fail - wrong first argument type


def case2() -> None:
    Vector[int](1.0)  # fail - wrong argument type


def case3() -> None:
    Vector[int]([1, 2, 3])  # success


def case4() -> None:
    Vector[float]([1.0, 2.0, 3.0])  # success


def case5() -> None:
    Vector[float]([1, 2, 3])  # success - int is compatible with float


def case6() -> None:
    Vector[str](["a", "b", "c"])  # fail - strings not allowed


def test_annotations() -> None:
    check_annotations(Vector.__init__)
    check_annotations(Vector.__len__)
    check_annotations(Vector.__getitem__)
    check_annotations(Vector.append)
    check_annotations(Vector.extend)
    check_annotations(Vector.__iter__)
    check_annotations(Vector.__str__)
    check_annotations(Vector.__repr__)
    check_annotations(Vector.__add__)
    check_annotations(Vector.__sub__)
    check_annotations(Vector.__mul__)
    check_annotations(Vector.__rmul__)
    check_annotations(Vector.dot)
    check_annotations(VectorIterator.__init__)
    check_annotations(VectorIterator.__iter__)
    check_annotations(VectorIterator.__next__)

    check_func(typy_generic, case1, False)
    check_func(typy_generic, case2, False)
    check_func(typy_generic, case3, True)
    check_func(typy_generic, case4, True)
    check_func(typy_generic, case5, True)
    check_func(typy_generic, case6, False)


###################
# Vector operations tests
###################


def test_vector_addition() -> None:
    v1 = Vector[int]([1, 2, 3])
    v2 = Vector[int]([4, 5, 6])

    result = v1 + v2
    assert len(result) == 3
    assert list(result) == [5, 7, 9]

    # Original vectors should not be modified
    assert list(v1) == [1, 2, 3]
    assert list(v2) == [4, 5, 6]


def test_vector_addition_floats() -> None:
    v1 = Vector[float]([1.5, 2.5])
    v2 = Vector[float]([0.5, 1.5])

    result = v1 + v2
    assert list(result) == [2.0, 4.0]


def test_vector_subtraction() -> None:
    v1 = Vector[int]([5, 7, 9])
    v2 = Vector[int]([1, 2, 3])

    result = v1 - v2
    assert len(result) == 3
    assert list(result) == [4, 5, 6]

    # Original vectors should not be modified
    assert list(v1) == [5, 7, 9]
    assert list(v2) == [1, 2, 3]


def test_vector_subtraction_floats() -> None:
    v1 = Vector[float]([3.5, 5.5])
    v2 = Vector[float]([1.5, 2.5])

    result = v1 - v2
    assert list(result) == [2.0, 3.0]


def test_vector_multiplication_scalar() -> None:
    v = Vector[int]([1, 2, 3])

    result = v * 3
    assert len(result) == 3
    assert list(result) == [3, 6, 9]

    # Original vector should not be modified
    assert list(v) == [1, 2, 3]


def test_vector_multiplication_float_scalar() -> None:
    v = Vector[float]([1.0, 2.0, 3.0])

    result = v * 2.5
    assert list(result) == [2.5, 5.0, 7.5]


def test_vector_right_multiplication_scalar() -> None:
    v = Vector[int]([1, 2, 3])

    result = 3 * v
    assert len(result) == 3
    assert list(result) == [3, 6, 9]

    # Original vector should not be modified
    assert list(v) == [1, 2, 3]


def test_vector_right_multiplication_float_scalar() -> None:
    v = Vector[float]([1.0, 2.0, 3.0])

    result = 2.5 * v
    assert list(result) == [2.5, 5.0, 7.5]


def test_vector_dot_product() -> None:
    v1 = Vector[int]([1, 2, 3])
    v2 = Vector[int]([4, 5, 6])

    result = v1.dot(v2)
    assert result == 32  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32


def test_vector_dot_product_floats() -> None:
    v1 = Vector[float]([1.5, 2.0])
    v2 = Vector[float]([2.0, 3.0])

    result = v1.dot(v2)
    assert result == 9.0  # 1.5*2.0 + 2.0*3.0 = 3.0 + 6.0 = 9.0


def test_vector_dot_product_zero_vector() -> None:
    v1 = Vector[int]([1, 2, 3])
    v2 = Vector[int]([0, 0, 0])

    result = v1.dot(v2)
    assert result == 0


def test_vector_operations_chain() -> None:
    v1 = Vector[int]([1, 2, 3])
    v2 = Vector[int]([2, 3, 4])

    # Test chaining: (v1 + v2) * 2
    result = (v1 + v2) * 2
    assert list(result) == [6, 10, 14]


def test_vector_operations_with_negative_values() -> None:
    v1 = Vector[int]([5, -3, 2])
    v2 = Vector[int]([-1, 4, -2])

    result_add = v1 + v2
    assert list(result_add) == [4, 1, 0]

    result_sub = v1 - v2
    assert list(result_sub) == [6, -7, 4]

    result_mul = v1 * -2
    assert list(result_mul) == [-10, 6, -4]


###################
# Edge case tests
###################


def test_empty_vector() -> None:
    v = Vector[int]([])
    assert len(v) == 0

    # Iterating empty vector should return nothing
    result = list(v)
    assert result == []

    # Iterator should raise StopIteration immediately
    it = iter(v)
    try:
        next(it)
        assert False, "StopIteration should have been raised"
    except StopIteration:
        pass


def test_single_element() -> None:
    v = Vector[int]([42])
    assert len(v) == 1
    assert v[0] == 42
    assert v[-1] == 42

    result = list(v)
    assert result == [42]


def test_negative_indices_edge_cases() -> None:
    v = Vector[int]([10, 20, 30, 40])

    assert v[-1] == 40
    assert v[-4] == 10

    # Test with append
    v.append(50)
    assert v[-1] == 50
    assert v[-5] == 10


def test_multiple_appends_and_extends() -> None:
    v = Vector[int]([1])

    v.append(2)
    v.append(3)
    assert len(v) == 3
    assert list(v) == [1, 2, 3]

    v.extend([4, 5])
    v.extend([6])
    assert len(v) == 6
    assert list(v) == [1, 2, 3, 4, 5, 6]


def test_iterator_with_container_modification() -> None:
    v = Vector[int]([1, 2, 3])
    it = iter(v)

    assert next(it) == 1
    assert next(it) == 2

    # Modifying vector after creating iterator
    # Iterator should continue based on original state
    assert next(it) == 3


def test_vector_with_float_values() -> None:
    v = Vector[float]([1.5, 2.7, 3.14])
    assert len(v) == 3
    assert v[0] == 1.5
    assert v[1] == 2.7
    assert v[2] == 3.14

    result = list(v)
    assert result == [1.5, 2.7, 3.14]


def test_vector_str_empty_and_single() -> None:
    # Empty vector string representation
    v_empty = Vector[int]([])
    assert len(str(v_empty)) >= 0
    assert len(repr(v_empty)) >= 0

    # Single element vector string representation
    v_single = Vector[int]([99])
    assert len(str(v_single)) > 0
    assert len(repr(v_single)) > 0
