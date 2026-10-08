"""Shared test setup.

`src/`, `smart_player/` and `board_reader/src/` are script-style packages rather
than installed ones, so every consumer puts them on `sys.path` itself. Doing it
here once means a new test file does not have to repeat the incantation -- and
it happens at collection time, before any test module is imported.

Note this file is *not* loaded when a test module is executed directly
(`python tests/test_x.py`), which is why the older test files keep their own
`sys.path` lines and their `__main__` runners.
"""

import os
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")

for _path in (
    _ROOT,
    os.path.join(_ROOT, "src"),
    os.path.join(_ROOT, "smart_player"),
    os.path.join(_ROOT, "board_reader", "src"),
):
    _abs = os.path.abspath(_path)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

import languages  # noqa: E402


@pytest.fixture(scope="session")
def pl():
    """The Polish language definition."""
    return languages.load("pl")


@pytest.fixture(scope="session", params=languages.available())
def spec(request):
    """Every language definition in turn, so a check written once applies to
    each language that gets added later without anybody remembering to."""
    return languages.load(request.param)
