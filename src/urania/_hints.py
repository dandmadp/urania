"""Friendly error-message helpers."""

import difflib


def did_you_mean(word: str, options) -> str:
    """" Did you mean 'x'?" for the closest option, or an empty string."""
    match = difflib.get_close_matches(str(word).lower(), [str(o).lower() for o in options], n=1, cutoff=0.6)
    if not match:
        return ""
    original = {str(o).lower(): o for o in options}[match[0]]
    return f" Did you mean {original!r}?"
