"""Regression test: `uv build` artifacts must not leak research data assets.

Builds both the sdist and wheel into a temporary directory (never the repo's
own `dist/`, to avoid a build-inside-build trap) and asserts the installable
package files are present while non-package historical research assets
(root `.nii`/`.npy` volumes, `samples/`, notebooks, docs site sources,
tests, agent/CI config) are absent from the redistributable package
artifact.
"""

from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROHIBITED_SUFFIXES = (".nii", ".npy", ".ipynb", ".png", ".yml", ".yaml")
PROHIBITED_PATH_FRAGMENTS = (
    "samples/",
    "notebooks/",
    "docs/",
    "tests/",
    ".github/",
    "site/",
)


def _build_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    subprocess.run(
        ["uv", "build", "--sdist", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sdists = list(tmp_path.glob("*.tar.gz"))
    wheels = list(tmp_path.glob("*.whl"))
    assert len(sdists) == 1, f"expected exactly one sdist, got {sdists}"
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return sdists[0], wheels[0]


def _assert_no_prohibited_members(names: list[str], archive_label: str) -> None:
    for name in names:
        assert not name.endswith(PROHIBITED_SUFFIXES), (
            f"{archive_label} contains prohibited research artifact: {name}"
        )
        assert not any(fragment in name for fragment in PROHIBITED_PATH_FRAGMENTS), (
            f"{archive_label} contains prohibited path: {name}"
        )


def test_sdist_excludes_data_assets_and_includes_package(tmp_path: Path) -> None:
    sdist_path, _wheel_path = _build_artifacts(tmp_path)

    with tarfile.open(sdist_path, "r:gz") as tf:
        names = tf.getnames()

    _assert_no_prohibited_members(names, "sdist")
    assert any(name.endswith("src/multimodal_ad/__init__.py") for name in names)
    assert any(name.endswith("pyproject.toml") for name in names)
    assert any(name.endswith("README.md") for name in names)
    assert any(name.endswith("LICENSE") for name in names)

    # 28 MB before this fix (mri1.nii + *.npy alone); keep a generous but
    # meaningful ceiling so a future accidental include still fails loudly.
    assert sdist_path.stat().st_size < 1_000_000, (
        f"sdist is {sdist_path.stat().st_size} bytes, expected a lean package-only archive"
    )


def test_wheel_excludes_data_assets_and_includes_package(tmp_path: Path) -> None:
    _sdist_path, wheel_path = _build_artifacts(tmp_path)

    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()

    _assert_no_prohibited_members(names, "wheel")
    assert any(name.endswith("multimodal_ad/__init__.py") for name in names)
    assert any(name.endswith(".dist-info/METADATA") for name in names)
