"""Structural checks for the thin notebooks under `notebooks/` (not `notebooks/legacy/`).

Parses `.ipynb` files as plain JSON (no `nbformat`/Jupyter dependency, so
these tests always run in the default `just test` job): every new notebook
must stay a thin orchestration layer over `multimodal_ad` -- no function/class
definitions, no forbidden Colab/shell/credential/absolute-path patterns, and
committed with cleared outputs (see AGENTS.md, "notebooks may orchestrate and
display only").

`notebooks/legacy/*.ipynb` are historical artifacts preserved byte-for-byte
(see `docs/legacy-notebooks-inventory.md`) and are intentionally excluded.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "google.colab",
    "drive.mount",
    "pip install",
    "pip3 install",
    "apt-get",
    "apt install",
    "/content/",
    "/Users/",
    "/home/",
    "C:\\Users",
    "api_key",
    "aws_secret",
    "password",
)


def _thin_notebooks() -> list[Path]:
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    assert notebooks, f"expected at least one notebook directly under {NOTEBOOKS_DIR}"
    return notebooks


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _cell_source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


@pytest.mark.parametrize("path", _thin_notebooks(), ids=lambda p: p.name)
def test_notebook_defines_no_functions_or_classes(path: Path) -> None:
    notebook = json.loads(path.read_text())
    for cell in _code_cells(notebook):
        source = _cell_source(cell)
        tree = ast.parse(source, filename=str(path))
        forbidden_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        assert not forbidden_nodes, (
            f"{path.name}: notebooks must only orchestrate/display multimodal_ad "
            f"APIs, no function/class definitions (found {forbidden_nodes[0].name!r})"
        )


@pytest.mark.parametrize("path", _thin_notebooks(), ids=lambda p: p.name)
def test_notebook_has_no_forbidden_patterns(path: Path) -> None:
    notebook = json.loads(path.read_text())
    for cell in _code_cells(notebook):
        source = _cell_source(cell)
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in source, (
                f"{path.name}: found forbidden pattern {pattern!r} (Colab mounts, shell "
                "installs, absolute paths, and credentials are not allowed in notebooks)"
            )


@pytest.mark.parametrize("path", _thin_notebooks(), ids=lambda p: p.name)
def test_notebook_outputs_are_cleared(path: Path) -> None:
    notebook = json.loads(path.read_text())
    for cell in _code_cells(notebook):
        assert cell.get("outputs") == [], f"{path.name}: code cell has uncleared outputs"
        assert cell.get("execution_count") is None, (
            f"{path.name}: code cell has a non-null execution_count"
        )


def test_legacy_notebooks_are_excluded_from_thinness_checks() -> None:
    legacy_dir = NOTEBOOKS_DIR / "legacy"
    legacy_notebooks = list(legacy_dir.glob("*.ipynb"))
    assert len(legacy_notebooks) == 6, (
        "expected the 5 legacy root notebooks plus images/3D Brain Plot.ipynb, "
        f"got {[p.name for p in legacy_notebooks]}"
    )
    thin_names = {p.name for p in _thin_notebooks()}
    assert thin_names.isdisjoint({p.name for p in legacy_notebooks})
