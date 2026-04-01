import pytest

##############
# Code section
##############


# Don't change this function!
def ctr(clicks: int, shows: int) -> float:
    """
    Calculate CTR. Contains bugs!

    :param clicks: number of clicks on banner (must be >= 0)
    :param shows: number of banner shows (must be > 0)
    :return: clicks-through rate.
             Bugs:
               - Returns int instead of float for some inputs
               - Raises ValueError for negative shows
               - Raises ZeroDivisionError for shows == 0
    """
    if shows < 0:
        raise ValueError("shows must be positive")
    return clicks // shows


def safe_ctr(clicks: int, shows: int, log: list[str]) -> float:
    """
    Safely calculate CTR, handling errors gracefully.

    Must use try, except, finally and assert:
    1. Inside the try block, assert that clicks is non-negative.
       If clicks < 0, raise AssertionError with message
       "Clicks must be non-negative".
    2. Call ctr(clicks, shows) and return the result as float.
    3. Catch ValueError (e.g. negative shows) and return -1.0.
    4. Catch ZeroDivisionError (shows == 0) and return 0.0.
    5. In the finally block, always append "done" to log.

    The finally block must execute even when assert raises
    an AssertionError that you do NOT catch.

    :param clicks: number of clicks on banner
    :param shows: number of banners shown
    :param log: list that must always receive a "done" entry
    :return: CTR as float, or -1.0 on ValueError, or 0.0 on
             ZeroDivisionError
    """


##############
# Test section
##############

@pytest.mark.skip
def test_check_ctr(clicks: int, shows: int, expected_result: float) -> None:
    """
    Write a simple test for the function ctr defined above.
    If ctr(clicks, shows) == expected_result, do nothing.
    Otherwise, raise AssertionError with the message
    "Wrong ctr calculation".

    :param clicks: parameter for ctr function
    :param shows: parameter for ctr function
    :param expected_result: result to compare with
    :return: None
    """
