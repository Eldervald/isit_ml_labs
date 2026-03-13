import io
import os
import sys
import testlib
import time

import pytest

from .context_manager import Timer, FileManager, OutputCapture


###################
# Structure asserts
###################


def test_docs() -> None:
    assert testlib.is_function_docstring_exists(Timer)
    assert testlib.is_function_docstring_exists(FileManager)
    assert testlib.is_function_docstring_exists(OutputCapture)


###################
# Timer tests
###################


def test_timer_basic() -> None:
    with Timer() as timer:
        time.sleep(0.01)
    assert timer.elapsed >= 0.01


def test_timer_elapsed_accessible() -> None:
    timer = Timer()
    with timer:
        time.sleep(0.005)
    elapsed = timer.elapsed
    assert elapsed >= 0.005
    assert elapsed < 1.0  # Should not be too long


def test_timer_returns_self() -> None:
    with Timer() as timer:
        assert isinstance(timer, Timer)
        assert hasattr(timer, '_start_time')


###################
# FileManager tests
###################


def test_file_manager_basic() -> None:
    filename = 'test_temp.txt'
    try:
        with FileManager(filename, 'w') as f:
            f.write('Hello, World!')

        with FileManager(filename, 'r') as f:
            content = f.read()
        assert content == 'Hello, World!'
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_file_manager_closes_on_exit() -> None:
    filename = 'test_temp2.txt'
    try:
        with FileManager(filename, 'w') as f:
            f.write('test')
        assert f.closed
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_file_manager_closes_on_exception() -> None:
    filename = 'test_temp3.txt'
    try:
        with FileManager(filename, 'w') as f:
            f.write('test')
            raise ValueError('test exception')
    except ValueError:
        pass
    assert f.closed
    if os.path.exists(filename):
        os.remove(filename)


def test_file_manager_read_write() -> None:
    filename = 'test_temp4.txt'
    try:
        # Write multiple lines
        with FileManager(filename, 'w') as f:
            f.write('Line 1\n')
            f.write('Line 2\n')

        # Read back
        with FileManager(filename, 'r') as f:
            lines = f.readlines()
        assert lines == ['Line 1\n', 'Line 2\n']
    finally:
        if os.path.exists(filename):
            os.remove(filename)


###################
# OutputCapture tests
###################


def test_output_capture_stdout() -> None:
    with OutputCapture() as captured:
        print('Hello')
        print('World')
    assert 'Hello' in captured.stdout
    assert 'World' in captured.stdout


def test_output_capture_stderr() -> None:
    with OutputCapture() as captured:
        print('Error message', file=sys.stderr)
    assert 'Error message' in captured.stderr


def test_output_capture_restores_streams() -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with OutputCapture() as captured:
        print('test')

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    # Also check that output was actually captured
    assert 'test' in captured.stdout


def test_output_capture_both_streams() -> None:
    with OutputCapture() as captured:
        print('stdout message')
        print('stderr message', file=sys.stderr)

    assert 'stdout message' in captured.stdout
    assert 'stderr message' in captured.stderr
    assert 'stderr message' not in captured.stdout


def test_output_capture_empty() -> None:
    with OutputCapture() as captured:
        pass
    assert captured.stdout == ''
    assert captured.stderr == ''


###################
# Edge cases
###################


def test_timer_no_operation() -> None:
    with Timer() as timer:
        pass
    assert timer.elapsed >= 0


def test_file_manager_default_mode() -> None:
    filename = 'test_temp5.txt'
    try:
        # Create file first
        with FileManager(filename, 'w') as f:
            f.write('test')

        # Open with default mode (read)
        with FileManager(filename) as f:
            content = f.read()
        assert content == 'test'
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_output_capture_nested() -> None:
    with OutputCapture() as outer:
        print('outer')
        with OutputCapture() as inner:
            print('inner')
        assert 'inner' in inner.stdout
        assert 'outer' not in inner.stdout
    assert 'outer' in outer.stdout
