"""No exception's text reaches a log, wherever it is caught.

A message can hold a student passage, a prompt, a provider's response body, a
connection string or a URL. Each path that logs an unexpected exception logs
its type and where it was raised (app.core.safe_log.describe), and attaches no
traceback, whose last line is the message again.
"""

from app.core.safe_log import describe

MARKER = "student-passage-5b8a"


def test_describe_names_the_type_and_place_never_the_text():
    try:
        raise ValueError(MARKER)
    except ValueError as exc:
        described = describe(exc)

    assert MARKER not in described
    assert described.startswith("ValueError at ")
    assert "test_safe_logging.py" in described
    assert describe(KeyError(MARKER)) == "KeyError"  # never raised: no place


class Leaky(Exception):
    """An exception whose own formatting holds the passage, and fails if used."""

    def __str__(self) -> str:
        raise AssertionError(f"describe called str(): {MARKER}")

    def __repr__(self) -> str:
        raise AssertionError(f"describe called repr(): {MARKER}")

    def __format__(self, spec: str) -> str:
        raise AssertionError(f"describe formatted the exception: {MARKER}")


def test_describe_never_formats_the_exception_or_its_cause():
    try:
        try:
            raise ValueError(MARKER)
        except ValueError as cause:
            raise Leaky(MARKER) from cause
    except Leaky as exc:
        described = describe(exc)

    assert described.startswith("Leaky at test_safe_logging.py:")
    assert MARKER not in described and "ValueError" not in described
