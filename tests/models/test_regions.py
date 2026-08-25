"""Tests for `multimodal_ad.models.regions` (no TensorFlow needed)."""

import numpy as np
import polars as pl
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
    # 4x4x4 = 64 voxels: the frame-count denominator used by `f"{name} mean"`.
    atlas = np.zeros((4, 4, 4), dtype=np.float64)
    atlas[0, 0, 0] = 10.0  # a single-voxel "region"
    atlas[1, 1, 1] = 20.0  # another single-voxel region

    heatmap = np.zeros((4, 4, 4), dtype=np.float64)
    heatmap[0, 0, 0] = 0.5
    heatmap[1, 1, 1] = 0.8

    labels = pl.DataFrame({"name": ["region_a", "region_b"], "intensity": [10.0, 20.0]})

    results = rank_regions(atlas, labels, {"pos": heatmap})

    row_a = results.filter(pl.col("part") == "region_a").row(0, named=True)
    row_b = results.filter(pl.col("part") == "region_b").row(0, named=True)
    # Single-voxel regions: frame mean and region mean coincide (sum / 64 vs
    # sum / 1 differ; see the multi-voxel test below for where they diverge).
    assert row_a["pos mean"] == pytest.approx(0.5 / 64)
    assert row_a["pos region mean"] == pytest.approx(0.5)
    assert row_a["pos count"] == 1
    assert row_a["pos sum"] == pytest.approx(0.5)
    assert row_b["pos mean"] == pytest.approx(0.8 / 64)
    assert row_b["pos region mean"] == pytest.approx(0.8)


def test_rank_regions_returns_nan_region_mean_for_absent_region() -> None:
    atlas = np.zeros((3, 3, 3), dtype=np.float64)  # no voxel has intensity 99
    heatmap = np.ones((3, 3, 3), dtype=np.float64)
    labels = pl.DataFrame({"name": ["missing_region"], "intensity": [99.0]})

    results = rank_regions(atlas, labels, {"neg": heatmap})
    row = results.row(0, named=True)
    # `mean` (frame-denominator formula) degrades gracefully to 0.0 for an
    # absent region (sum 0 / fixed frame count); `region mean` has no
    # region voxels to divide by, so it is `nan` rather than a fabricated 0.
    assert row["neg mean"] == pytest.approx(0.0)
    assert np.isnan(row["neg region mean"])
    assert row["neg count"] == 0


def test_rank_regions_mean_and_region_mean_diverge_for_different_sized_regions() -> None:
    """Distinguishes the two mean formulas: legacy `mean` dilutes by frame
    size (so a bigger region's `mean` looks smaller for the same per-voxel
    signal), while `region mean` is comparable across region sizes.
    """
    # 5x5x4 = 100 voxels: a round frame-count denominator for easy assertions.
    atlas = np.zeros((5, 5, 4), dtype=np.float64)
    atlas[0, 0, 0] = 10.0  # region_small: 1 voxel
    atlas[0:2, 0:2, 1] = 20.0  # region_big: 4 voxels

    heatmap = np.zeros((5, 5, 4), dtype=np.float64)
    heatmap[0, 0, 0] = 1.0  # region_small: sum 1.0
    heatmap[0:2, 0:2, 1] = 1.0  # region_big: sum 4.0, same per-voxel value

    labels = pl.DataFrame({"name": ["region_small", "region_big"], "intensity": [10.0, 20.0]})

    results = rank_regions(atlas, labels, {"pos": heatmap})
    small = results.filter(pl.col("part") == "region_small").row(0, named=True)
    big = results.filter(pl.col("part") == "region_big").row(0, named=True)

    # Legacy `mean`: sum / 100 voxels regardless of region size, so the
    # 4-voxel region's frame mean is 4x the 1-voxel region's, purely from
    # having a bigger sum, not a bigger per-voxel signal.
    assert small["pos mean"] == pytest.approx(1.0 / 100)
    assert big["pos mean"] == pytest.approx(4.0 / 100)
    # `region mean`: sum / own voxel count, identical per-voxel signal in
    # both regions gives identical region means (1.0), unlike `mean` above.
    assert small["pos region mean"] == pytest.approx(1.0)
    assert big["pos region mean"] == pytest.approx(1.0)


def test_atlas_and_heatmap_placements_are_disjoint_within_common_frame() -> None:
    # Both placements land inside the same (128, 128, 128) frame; sanity check
    # that pad_to_frame accepts both retained offsets without raising.
    heat_frame = pad_to_frame(np.zeros((128, 128, 50)), HEATMAP_PLACEMENT)
    atlas_frame = pad_to_frame(np.zeros((91, 109, 91)), ATLAS_PLACEMENT)
    assert heat_frame.shape == atlas_frame.shape == (128, 128, 128)
