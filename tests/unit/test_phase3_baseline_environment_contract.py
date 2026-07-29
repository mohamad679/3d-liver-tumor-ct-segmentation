"""Static regression tests for the isolated Phase 3 baseline test runner."""

from __future__ import annotations

import tomllib
from pathlib import Path

_BASELINE_PROJECT = Path("environments/phase3-baselines/intel-macos-cpu")
_ROOT_PYTEST_VERSION = "9.1.1"


def _make_target_recipe(target_name: str) -> str:
    """Return one Makefile target recipe without inspecting user-specific paths."""

    lines = Path("Makefile").read_text(encoding="utf-8").splitlines()
    target_index = lines.index(f"{target_name}:")
    recipe_lines: list[str] = []
    for line in lines[target_index + 1 :]:
        if line and not line.startswith("\t"):
            break
        if line.startswith("\t"):
            recipe_lines.append(line.removeprefix("\t"))
    if not recipe_lines:
        raise AssertionError(f"Makefile target {target_name!r} has no recipe.")
    return "\n".join(recipe_lines)


def test_phase3_baseline_test_runner_contract_is_interpreter_safe() -> None:
    """Require the baseline pytest pin and interpreter-safe locked Makefile command."""

    payload = tomllib.loads((_BASELINE_PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["dependency-groups"]["dev"] == [f"pytest=={_ROOT_PYTEST_VERSION}"]

    recipe = _make_target_recipe("baseline-test")
    assert "python -m pytest" in recipe
    assert recipe.count("pytest") == 1
    assert f"--project {_BASELINE_PROJECT.as_posix()}" in recipe
    assert "--locked" in recipe
    assert "--no-sync" in recipe
