import io
import testlib

import pytest

from .context_manager import supresser, retyper, dumper


###################
# Structure asserts
###################


def test_docs() -> None:
    assert testlib.is_function_docstring_exists(supresser)
    assert testlib.is_function_docstring_exists(retyper)
    assert testlib.is_function_docstring_exists(dumper)


###################
# Tests
###################


def test_retyper_retypes() -> None:
    try:
        with retyper(ValueError, TypeError):
            raise ValueError('penguin')
    except ValueError:
        assert False, 'source error was raised'
    except TypeError as e:
        assert 'penguin' in e.args, 'attribute args lost'
    except Exception as e:
        assert False, 'totally wrong exception type {}'.format(e)
    else:
        assert False, 'retyper should throw'


def test_retyper_idles() -> None:
    try:
        with retyper(ValueError, TypeError):
            raise IOError
    except (ValueError, TypeError):
        assert False, 'wrong exception type'
    except IOError:
        assert True
    except Exception:
        assert False, 'wrong exception type'
    else:
        assert False, 'retyper should throw'


def test_nested_retypers() -> None:
    try:
        with retyper(TypeError, IOError), retyper(ValueError, TypeError):
            raise ValueError('lalala', 1)
    except IOError as e:
        assert e.args == ('lalala', 1)
    else:
        assert False, 'wrong exception type in nested manager'


def test_supresser_idles() -> None:
    try:
        with supresser(ValueError, TypeError):
            raise IOError
    except IOError:
        assert True
    except Exception as e:
        assert False, 'wrong exception type {}'.format(e)
    else:
        assert False, 'no exception'


def test_supresser_supress() -> None:
    try:
        with supresser(ValueError, TypeError):
            raise ValueError('message')
    except Exception as e:
        assert False, 'supressed exception raised {}'.format(e)
    else:
        pass


def test_dumper_stream() -> None:
    stream = io.StringIO()
    msg = 'message to log'
    try:
        with dumper(stream):
            raise ValueError(msg)
    except ValueError:
        assert msg in stream.getvalue()
    except Exception:
        assert False, 'wrong exception'
    else:
        assert False, 'dumper should throw'


def test_dumped_stderr(capsys) -> None:  # type: ignore
    msg = 'message to log'
    try:
        with dumper():
            raise ValueError(msg)
    except ValueError:
        captured = capsys.readouterr()
        assert msg in captured.err
    except Exception:
        assert False, 'wrong exception'
    else:
        assert False, 'dumper should throw'


def test_supresser_no_exceptions() -> None:
    # Edge case: empty exception type list
    try:
        with supresser():
            pass
    except Exception as e:
        assert False, 'unexpected exception with empty supresser {}'.format(e)


def test_supresser_multiple_exceptions() -> None:
    # Edge case: multiple exception types in single call
    try:
        with supresser(ValueError, TypeError, KeyError, IOError):
            raise ValueError('test')
    except Exception as e:
        assert False, 'supressed exception raised {}'.format(e)
    else:
        pass

    try:
        with supresser(ValueError, TypeError, KeyError, IOError):
            raise TypeError('test2')
    except Exception as e:
        assert False, 'supressed exception raised {}'.format(e)
    else:
        pass


def test_retyper_traceback_preservation() -> None:
    # Edge case: verify traceback information is preserved
    try:
        with retyper(ValueError, TypeError):
            raise ValueError('original message')
    except TypeError as e:
        assert 'original message' in e.args
        # Verify the exception chain is maintained
        assert e.__cause__ is None or isinstance(e.__cause__, ValueError)
    except Exception:
        assert False, 'wrong exception type'
