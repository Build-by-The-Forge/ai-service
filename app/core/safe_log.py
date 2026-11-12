"""What may be logged about an exception: its type and where it was raised.

Never its text. A message can hold a student passage, a prompt, a provider's
response body, a connection string or a URL, none of which belongs in a log.
Nor a traceback, whose last line is the message again.
"""

import os
import traceback


def describe(exc: BaseException) -> str:
    """The exception's type and the frame that raised it, such as
    "TimeoutError at fetcher.py:12 in fetch".

    Only controlled metadata: the class name, and the raising frame's file
    name, line and function from the traceback. It never calls str() or
    repr() on the exception, reads none of its attributes or arguments, and
    ignores chained exceptions, so a custom __str__ can't leak through it."""
    name = type(exc).__name__
    frames = traceback.extract_tb(exc.__traceback__)  # empty when it has none
    if not frames:
        return name
    last = frames[-1]
    return f"{name} at {os.path.basename(last.filename)}:{last.lineno} in {last.name}"
