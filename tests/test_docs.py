"""Check that the README python blocks and the example scripts actually run."""

import re
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)


@pytest.mark.parametrize("index", range(len(_python_blocks())))
def test_readme_block_runs(index):
    exec(compile(_python_blocks()[index], f"README.md[{index}]", "exec"), {})
    plt.close("all")


def test_readme_quickstart_numbers():
    """Numbers written in the README must match the actual results."""
    import urania as ur

    text = README.read_text(encoding="utf-8")
    assert repr(ur.ISS.transfer_to(ur.GEO)) in text


@pytest.mark.parametrize("script", ["quickstart.py", "tle_vs_propagator.py"])
def test_example_runs(script, tmp_path, monkeypatch):
    # Move __file__ so the figure is saved in a temporary folder
    src = (ROOT / "examples" / script).read_text(encoding="utf-8")
    target = tmp_path / script
    target.write_text(src, encoding="utf-8")
    runpy.run_path(str(target), run_name="__main__")
    assert target.with_suffix(".png").exists()
    plt.close("all")
