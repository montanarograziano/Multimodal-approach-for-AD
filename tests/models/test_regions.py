"""Tests for `multimodal_ad.models.regions` (no TensorFlow needed)."""

import numpy as np
import pandas as pd
import pytest

from multimodal_ad.models.regions import (
    ATLAS_PLACEMENT,
    HEATMAP_PLACEMENT,
    pad_to_frame,
    rank_regions,
)


def test_pad_to_frame_places_volume_at_the_expected_offset() -> None:
    heatmap = np.ones((128, 128, 50), dtype=np.float64)
    frame = pad_to_frame(heatmap, HEATMAP_PLACEMENT)

    assert frame.shape == (128, 128, 128)
    assert np.all(frame[:, :, :39] == 0.0)
    assert np.all(frame[:, :, 39:89] == 1.0)
    assert np.all(frame[:, :, 89:] == 0.0)


def test_pad_to_frame_rejects_mismatched_shape() -> None:
    heatmap = np.ones((10, 10, 10), dtype=np.float64)
    with pytest.raises(ValueError, match="does not match placement extent"):
        pad_to_frame(heatmap, HEATMAP_PLACEMENT)


def test_rank_regions_computes_masked_mean_count_sum() -> None:
    atlas = np.zeros((4, 4, 4), dtype=np.float64)
    atlas[0, 0, 0] = 10.0  # a single-voxel "region"
    atlas[1, 1, 1] = 20.0  # another single-voxel region

    heatmap = np.zeros((4, 4, 4), dtype=np.float64)
    heatmap[0, 0, 0] = 0.5
    heatmap[1, 1, 1] = 0.8

    labels = pd.DataFrame({"intensity": [10.0, 20.0]}, index=["region_a", "region_b"])
    labels.index.name = "name"

    results = rank_regions(atlas, labels, {"pos": heatmap})

    row_a = results[results["part"] == "region_a"].iloc[0]
    row_b = results[results["part"] == "region_b"].iloc[0]
    assert row_a["pos mean"] == pytest.approx(0.5)
    assert row_a["pos count"] == 1
    assert row_a["pos sum"] == pytest.approx(0.5)
    assert row_b["pos mean"] == pytest.approx(0.8)


def test_rank_regions_returns_nan_mean_for_absent_region() -> None:
    atlas = np.zeros((3, 3, 3), dtype=np.float64)  # no voxel has intensity 99
    heatmap = np.ones((3, 3, 3), dtype=np.float64)
    labels = pd.DataFrame({"intensity": [99.0]}, index=["missing_region"])

    results = rank_regions(atlas, labels, {"neg": heatmap})
    assert np.isnan(results.iloc[0]["neg mean"])
    assert results.iloc[0]["neg count"] == 0


def test_atlas_and_heatmap_placements_are_disjoint_within_common_frame() -> None:
    # Both placements land inside the same (128, 128, 128) frame; sanity check
    # that pad_to_frame accepts both retained offsets without raising.
    heat_frame = pad_to_frame(np.zeros((128, 128, 50)), HEATMAP_PLACEMENT)
    atlas_frame = pad_to_frame(np.zeros((91, 109, 91)), ATLAS_PLACEMENT)
    assert heat_frame.shape == atlas_frame.shape == (128, 128, 128)
