"""Integrity check for the bundled AAL2 atlas asset (no TensorFlow needed).

Guards the repository-root `atlas.nii.gz` + `AAL2_Atlas_Labels.csv` pair
that `multimodal_ad.models.regions` depends on: wrong shape/voxel size or a
label mismatch would silently break region ranking without failing loudly
elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest

from multimodal_ad.models.regions import load_atlas, load_region_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
ATLAS_PATH = REPO_ROOT / "atlas.nii.gz"
LABELS_PATH = REPO_ROOT / "AAL2_Atlas_Labels.csv"


def test_atlas_file_exists() -> None:
    assert ATLAS_PATH.is_file(), f"expected bundled atlas at {ATLAS_PATH}"


def test_atlas_shape_and_voxel_size() -> None:
    image = cast(nib.spatialimages.SpatialImage, nib.load(ATLAS_PATH))
    assert image.shape == (91, 109, 91)
    assert tuple(image.header.get_zooms()) == pytest.approx((2.0, 2.0, 2.0))


def test_nonzero_intensities_match_labels_csv() -> None:
    atlas = load_atlas(ATLAS_PATH)
    labels = load_region_labels(str(LABELS_PATH))

    nonzero_intensities = sorted(float(v) for v in np.unique(atlas) if v != 0)
    csv_intensities = sorted(float(v) for v in labels["intensity"])

    assert len(nonzero_intensities) == 120
    assert nonzero_intensities == csv_intensities


def test_load_atlas_default_path_resolves_from_repo_root() -> None:
    # Mirrors the documented default: `load_atlas()` with no args, run from
    # the repo root, as notebooks/tooling launched from there would.
    original_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        atlas = load_atlas()
        assert atlas.shape == (91, 109, 91)
    finally:
        os.chdir(original_cwd)
