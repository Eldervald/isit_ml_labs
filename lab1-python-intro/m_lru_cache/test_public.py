from typing import Any, TypeVar

from _pytest.capture import CaptureFixture  # typing
import testlib

from .lru_cache import cache


###################
# Structure asserts
###################


def test_docs() -> None:
    assert testlib.is_function_docstring_exists(cache)


###################
# Tests
###################


@cache(20)
def binomial(n: int, k: int) -> int:
    if k > n:
        return 0
    if k == 0:
        return 1
    return binomial(n - 1, k) + binomial(n - 1, k - 1)


@cache(2048)
def ackermann(m: int, n: int) -> int:
    print(f'Calculating for {m} and {n}...')
    if m == 0:
        return n + 1
    if m > 0 and n == 0:
        return ackermann(m - 1, 1)
    if m > 0 and n > 0:
        return ackermann(m - 1, ackermann(m, n - 1))
    assert False, 'unreachable'


@cache(1)
def join(args: tuple[Any, ...] | Any) -> tuple[Any, ...]:
    result: tuple[Any, ...] = tuple()
    for arg in args:
        if isinstance(arg, tuple):
            result += join(arg)
        else:
            result += (arg,)
    return result


def test_cache_not_changes_func() -> None:
    T = TypeVar('T')

    @cache(1)
    def func(a: T) -> T:
        """test doc"""
        return a

    assert func.__name__ == 'func'
    assert func.__doc__ == 'test doc'
    assert func.__module__ == __name__


def test_binomial() -> None:
    result = sum(binomial(30, i) for i in range(31))
    assert result == 2 ** 30


def test_ackermann(capsys: CaptureFixture[str]) -> None:
    result = ackermann(3, 7)
    assert result == 1021
    assert capsys.readouterr().out.count('\n') == 2558


def test_join_lists() -> None:
    result = join(((1, 2, 3), 1, (1, 2, 3, 4, ((1, 2), 2, 3))))
    assert result == (1, 2, 3, 1, 1, 2, 3, 4, 1, 2, 2, 3)


def test_max_cache_size() -> None:
    calls_count = 0
    cache_size = 8

    @cache(cache_size)
    def simple_id(i: int) -> int:
        nonlocal calls_count
        calls_count += 1
        return i

    args = tuple(range(cache_size))
    result = tuple(map(simple_id, args))
    assert result == args
    assert calls_count == cache_size

    args = tuple(range(cache_size, cache_size * 2))
    result = tuple(map(simple_id, args))
    assert result == args
    assert calls_count == cache_size * 2

    args = tuple(range(cache_size))
    result = tuple(map(simple_id, args))
    assert result == args
    assert calls_count == cache_size * 3


###################
# Additional edge case tests
###################


def test_cache_size_zero() -> None:
    # Edge case: cache size of 0
    calls_count = 0

    @cache(0)
    def always_compute(i: int) -> int:
        nonlocal calls_count
        calls_count += 1
        return i * 2

    # With cache size 0, every call should compute
    assert always_compute(1) == 2
    assert calls_count == 1

    assert always_compute(1) == 2
    assert calls_count == 2

    assert always_compute(2) == 4
    assert calls_count == 3


def test_cache_keyword_arguments() -> None:
    # Edge case: keyword arguments support
    calls_count = 0

    @cache(10)
    def func_with_kwargs(a: int, b: int = 5) -> int:
        nonlocal calls_count
        calls_count += 1
        return a + b

    # Call with positional args
    assert func_with_kwargs(1, 2) == 3
    assert calls_count == 1

    # Call with keyword args
    assert func_with_kwargs(1, b=2) == 3
    assert calls_count == 1  # Should use cached result

    # Call with default value
    assert func_with_kwargs(1) == 6
    assert calls_count == 2

    assert func_with_kwargs(1, b=5) == 6
    assert calls_count == 2  # Should use cached result


def test_cache_empty_argument_function() -> None:
    # Edge case: function with no arguments
    calls_count = 0

    @cache(5)
    def no_args() -> int:
        nonlocal calls_count
        calls_count += 1
        return 42

    assert no_args() == 42
    assert calls_count == 1

    assert no_args() == 42
    assert calls_count == 1  # Should use cached result


def test_cache_size_one_eviction() -> None:
    # Edge case: cache size of 1 with eviction
    calls_count = 0

    @cache(1)
    def single_slot(i: int) -> int:
        nonlocal calls_count
        calls_count += 1
        return i * 10

    # First call fills cache
    assert single_slot(1) == 10
    assert calls_count == 1

    # Second call should hit cache
    assert single_slot(1) == 10
    assert calls_count == 1

    # Different argument evicts previous entry
    assert single_slot(2) == 20
    assert calls_count == 2

    # First argument is no longer cached
    assert single_slot(1) == 10
    assert calls_count == 3

    # Second argument is also evicted
    assert single_slot(2) == 20
    assert calls_count == 4
