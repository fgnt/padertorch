import sys
import pytest

def skip_on_windows():
    """Skip test if the current platform is Windows.

    This is necessary in the tests for the training.
    """
    if sys.platform.startswith('win'):
        pytest.skip('this doctest does not work on Windows, '
                    'training is not possible on Windows because file symlinks are unavailable for non-admin users')