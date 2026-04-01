import pytest
import testlib

from .error_handling import test_check_ctr, safe_ctr


###################
# Structure asserts
###################


def test_docs() -> None:
    assert testlib.is_function_docstring_exists(test_check_ctr)
    assert testlib.is_function_docstring_exists(safe_ctr)


def test_safe_ctr_uses_try() -> None:
    assert testlib.is_bytecode_op_used(safe_ctr, 'PUSH_EXC_INFO'), \
        'safe_ctr should use try block'


def test_safe_ctr_uses_specific_except() -> None:
    assert testlib.is_bytecode_op_used(safe_ctr, 'CHECK_EXC_MATCH'), \
        'safe_ctr should catch specific exception types, not bare except'


def test_safe_ctr_catches_zerodivisionerror() -> None:
    assert testlib.is_global_used(safe_ctr, 'ZeroDivisionError'), \
        'safe_ctr should catch ZeroDivisionError'


def test_safe_ctr_uses_assert() -> None:
    assert testlib.is_bytecode_op_used(safe_ctr, 'LOAD_ASSERTION_ERROR'), \
        'safe_ctr should use assert keyword for precondition check'


def test_test_check_ctr_uses_ctr() -> None:
    assert testlib.is_global_used(test_check_ctr, 'ctr'), \
        'test_check_ctr should call ctr function'


def test_test_check_ctr_uses_assert() -> None:
    assert testlib.is_bytecode_op_used(test_check_ctr, 'LOAD_ASSERTION_ERROR'), \
        'test_check_ctr should use assert keyword'


###################
# safe_ctr functional tests
###################


def test_safe_ctr_normal_case() -> None:
    log: list[str] = []
    result = safe_ctr(2, 2, log)
    assert isinstance(result, float)
    assert result == 1.0
    assert log == ['done']


def test_safe_ctr_fractional_bug() -> None:
    log: list[str] = []
    result = safe_ctr(1, 2, log)
    assert isinstance(result, float)
    assert result == 0.0
    assert log == ['done']


def test_safe_ctr_zero_shows_catches_zerodivision() -> None:
    log: list[str] = []
    result = safe_ctr(10, 0, log)
    assert isinstance(result, float)
    assert result == 0.0
    assert log == ['done']


def test_safe_ctr_negative_shows_catches_valueerror() -> None:
    log: list[str] = []
    result = safe_ctr(5, -1, log)
    assert isinstance(result, float)
    assert result == -1.0
    assert log == ['done']


def test_safe_ctr_negative_clicks_assert_fires() -> None:
    """Assert fires for negative clicks, but finally still runs."""
    log: list[str] = []
    with pytest.raises(AssertionError, match="Clicks must be non-negative"):
        safe_ctr(-1, 5, log)
    assert 'done' in log, \
        'finally block must execute even when an uncaught exception propagates'


def test_safe_ctr_finally_on_assert() -> None:
    """Verify finally appends 'done' exactly once even when assert fires."""
    log: list[str] = []
    try:
        safe_ctr(-3, 10, log)
    except AssertionError:
        pass
    assert log.count('done') == 1, \
        'finally block should append "done" exactly once'


def test_safe_ctr_return_type_is_float() -> None:
    log: list[str] = []
    result = safe_ctr(3, 4, log)
    assert isinstance(result, float)
    assert result == 0.0
    assert log == ['done']


###################
# test_check_ctr functional tests
###################


def test_check_ctr_correct_value() -> None:
    test_check_ctr(2, 2, 1.0)


def test_check_ctr_zero_clicks() -> None:
    test_check_ctr(0, 100, 0.0)


def test_check_ctr_fractional_catches_bug() -> None:
    with pytest.raises(AssertionError, match="Wrong ctr calculation"):
        test_check_ctr(1, 2, 0.5)


def test_check_ctr_clicks_gt_shows_catches_bug() -> None:
    with pytest.raises(AssertionError, match="Wrong ctr calculation"):
        test_check_ctr(10, 5, 1.0)
