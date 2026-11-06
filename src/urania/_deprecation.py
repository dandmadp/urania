"""Deprecation helpers (see docs/stability.md).

A public name that changes keeps working for at least one minor release and warns with
UraniaDeprecationWarning, a FutureWarning so that end users see it by default.
"""

from __future__ import annotations

import functools
import warnings


class UraniaDeprecationWarning(FutureWarning):
    """A urania feature that will be removed or renamed; the message says what to use instead."""


def warn_deprecated(what: str, *, since: str, removed_in: str, instead: str = "", stacklevel: int = 3) -> None:
    """Issue the standard deprecation warning."""
    tail = f"; use {instead} instead" if instead else ""
    warnings.warn(f"{what} is deprecated since urania {since} and will be removed in {removed_in}{tail}",
                  UraniaDeprecationWarning, stacklevel=stacklevel)


def deprecated(*, since: str, removed_in: str, instead: str = ""):
    """Decorator for a deprecated function or method."""
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(f"{func.__qualname__}()", since=since, removed_in=removed_in, instead=instead)
            return func(*args, **kwargs)

        wrapper.__doc__ = (f"Deprecated since {since}, removed in {removed_in}"
                           + (f"; use {instead}" if instead else "") + ".\n\n" + (func.__doc__ or ""))
        return wrapper

    return decorate
