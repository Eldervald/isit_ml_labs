import io
import sys
import time
from typing import TextIO


class Timer:
    """Context manager for measuring execution time."""

    def __init__(self):
        self._start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FileManager:
    """Context manager for file operations that ensures proper cleanup."""

    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class OutputCapture:
    """Context manager that captures stdout and stderr output."""

    def __init__(self):
        self._old_stdout = None
        self._old_stderr = None
        self.stdout = ""
        self.stderr = ""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Примеры использования (для самопроверки после реализации):
#
# import os
# import sys
#
# # Timer
# with Timer() as timer:
#     time.sleep(0.1)
# print(f"Elapsed: {timer.elapsed:.2f}s")
#
# # FileManager
# with FileManager('demo.txt', 'w') as f:
#     f.write('Hello!')
# with FileManager('demo.txt', 'r') as f:
#     print(f.read())
# os.remove('demo.txt')
#
# # OutputCapture
# with OutputCapture() as captured:
#     print("Test output")
# print(f"Captured: {captured.stdout}")
#
# # Вложенные контекстные менеджеры
# with Timer() as t:
#     with OutputCapture() as cap:
#         print("Nested!")
# print(f"Time: {t.elapsed:.2f}s, Output: {cap.stdout.strip()}")
