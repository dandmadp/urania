"""The public API is frozen in tests/fixtures/public_api.txt (see docs/stability.md).

A change to a public name or signature fails this test. When the change is intended, regenerate the snapshot:

    URANIA_UPDATE_API=1 .venv/Scripts/python -m pytest tests/test_api_stability.py

and describe the change (with deprecations for removals) in CHANGELOG.md.
"""

import importlib
import inspect
import os
import warnings
from pathlib import Path

import pytest

import urania
from urania._deprecation import UraniaDeprecationWarning, deprecated, warn_deprecated

SNAPSHOT = Path(__file__).parent / "fixtures" / "public_api.txt"
CORE_MODULES = ("coverage", "elements", "flyby", "forces", "frames", "interplanetary", "j2", "kepler", "lambert",
                "lifetime", "maneuvers", "planets", "propagate", "propulsion", "relative", "sun", "thirdbody",
                "twobody")


def _signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def _describe_class(name: str, cls) -> list[str]:
    lines = [f"class {name}{_signature(cls)}"]
    for attr, value in sorted(vars(cls).items()):
        if attr.startswith("_"):
            continue
        if isinstance(value, (property,)) or type(value).__name__ == "cached_property":
            lines.append(f"  {name}.{attr} [property]")
        elif isinstance(value, (classmethod, staticmethod)):
            lines.append(f"  {name}.{attr}{_signature(getattr(cls, attr))}")
        elif callable(value):
            lines.append(f"  {name}.{attr}{_signature(value)}")
    fields = getattr(cls, "__dataclass_fields__", None) or getattr(cls, "_fields", None)
    if fields:
        lines.append(f"  {name} fields: {', '.join(fields)}")
    return lines


def public_api() -> list[str]:
    lines = []
    for name in sorted(urania.__all__):
        obj = getattr(urania, name)
        if inspect.isclass(obj):
            lines += _describe_class(name, obj)
        elif callable(obj):
            lines.append(f"def {name}{_signature(obj)}")
        else:
            lines.append(f"{name}: {type(obj).__name__}")
    for mod_name in CORE_MODULES:
        mod = importlib.import_module(f"urania.core.{mod_name}")
        for name, obj in sorted(vars(mod).items()):
            if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isfunction(obj):
                lines.append(f"def core.{mod_name}.{name}{_signature(obj)}")
            elif inspect.isclass(obj):
                lines += _describe_class(f"core.{mod_name}.{name}", obj)
    return lines


def test_public_api_matches_snapshot():
    current = public_api()
    if os.environ.get("URANIA_UPDATE_API"):
        SNAPSHOT.write_text("\n".join(current) + "\n", encoding="utf-8")
    expected = SNAPSHOT.read_text(encoding="utf-8").splitlines()
    removed = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    assert not removed and not added, (
        "The public API changed.\nRemoved or changed:\n  " + "\n  ".join(removed) + "\nAdded:\n  " + "\n  ".join(added)
        + "\nIf intended: URANIA_UPDATE_API=1 pytest tests/test_api_stability.py, and note it in CHANGELOG.md")


def test_deprecation_helpers():
    @deprecated(since="0.1", removed_in="0.2", instead="new_thing()")
    def old_thing(x):
        """Old docs."""
        return 2 * x

    with pytest.warns(UraniaDeprecationWarning, match=r"old_thing\(\) is deprecated since urania 0.1.*new_thing"):
        assert old_thing(3) == 6
    assert "Deprecated since 0.1" in old_thing.__doc__ and "Old docs." in old_thing.__doc__
    assert issubclass(UraniaDeprecationWarning, FutureWarning)       # shown to end users by default
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated("x", since="0.1", removed_in="0.2", stacklevel=2)
    assert caught[0].filename == __file__
