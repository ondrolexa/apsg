

# No tests for paleomag -- OMG!
# At least test that module is properly imported.
# :( Refactoring of Python is horrble, horble, horrible experience!


import pytest


def test_import():
    from apsg.paleomag import Core
    assert Core is not None