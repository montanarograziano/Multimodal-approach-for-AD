"""Tests for `multimodal_ad.data.volumes`."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from multimodal_ad.data.manifest import Modality
from multimodal_ad.data.volumes import (
    MRI_BRAIN_BOX_PARAMS,
    BoundingBox,
    BrainBoxParams,
    ProcessingConfig,
    average_4d_frames,
    central_axial_slices,
    crop_and_resize_frame,
    find_brain_bounding_box,
    load_volume,
    normalize_intensity,
    process_scan,
    process_volume,
)


def _brain_like_volume(shape: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    """A noisy background with a brighter cube spanning the full depth axis.

    Spans all Z slices (unlike a small isolated blob) so every axial slice
    has real foreground to detect, matching a real head volume where brain
    tissue is present in every slice of the processed central region.
    """
    rng = np.random.default_rng(0)
    volume = rng.normal(50, 5, shape)
    lo, hi = np.array(shape) // 4, np.array(shape) - np.array(shape) // 4
    blob_shape = (hi[0] - lo[0], hi[1] - lo[1], shape[2])
    volume[lo[0] : hi[0], lo[1] : hi[1], :] = rng.normal(500, 50, blob_shape)
    return np.clip(volume, 0, None).astype(np.float32)


def test_average_4d_frames_collapses_frame_axis() -> None:
    volume = np.stack([np.full((4, 4, 4), 1.0), np.full((4, 4, 4), 3.0)], axis=-1)
    averaged = average_4d_frames(volume)
    assert averaged.shape == (4, 4, 4)
    assert np.allclose(averaged, 2.0)


def test_average_4d_frames_passes_through_3d() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float64)
    averaged = average_4d_frames(volume)
    assert averaged.shape == (4, 4, 4)
    assert averaged.dtype == np.float32


def test_average_4d_frames_rejects_bad_ndim() -> None:
    with pytest.raises(ValueError, match="3D or 4D"):
        average_4d_frames(np.ones((4, 4)))


def test_normalize_intensity_scales_to_unit_range() -> None:
    volume = np.array([0.0, 5.0, 10.0], dtype=np.float32)
    normalized = normalize_intensity(volume)
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
    assert normalized.dtype == np.float32


def test_normalize_intensity_handles_constant_volume() -> None:
    volume = np.full((4, 4), 7.0)
    normalized = normalize_intensity(volume)
    assert np.all(normalized == 0.0)


def test_central_axial_slices_extracts_middle() -> None:
    volume = np.arange(10).reshape(1, 1, 10)
    central = central_axial_slices(volume, 4)
    assert central.shape == (1, 1, 4)
    assert central.flatten().tolist() == [3, 4, 5, 6]


def test_central_axial_slices_rejects_too_many_frames() -> None:
    volume = np.zeros((1, 1, 5))
    with pytest.raises(ValueError, match="exceeds volume depth"):
        central_axial_slices(volume, 10)


def test_find_brain_bounding_box_detects_central_blob() -> None:
    from multimodal_ad.data.volumes import MRI_BRAIN_BOX_PARAMS

    # The legacy MRI Otsu params (51x51 Gaussian kernel) are tuned for
    # ~128px frames; a much smaller frame gets over-smoothed into a
    # near-uniform image with no detectable contour. Use a realistically
    # sized volume so the detector has a real bright region to find.
    volume = _brain_like_volume((128, 128, 16))
    box = find_brain_bounding_box(volume, MRI_BRAIN_BOX_PARAMS)
    assert isinstance(box, BoundingBox)
    # The detected box should roughly cover the central bright region, not
    # the whole (128, 128, ...) frame.
    assert 0 <= box.x < 60
    assert box.width < 128


def test_find_brain_bounding_box_falls_back_on_degenerate_slice() -> None:
    from multimodal_ad.data.volumes import MRI_BRAIN_BOX_PARAMS

    volume = np.zeros((16, 16, 3), dtype=np.float32)  # fully flat: no contour possible
    box = find_brain_bounding_box(volume, MRI_BRAIN_BOX_PARAMS)
    assert box.width > 0 and box.height > 0


def test_crop_and_resize_frame_output_shape() -> None:
    frame = np.zeros((32, 32), dtype=np.float32)
    box = BoundingBox(x=4, y=4, width=8, height=12)
    resized = crop_and_resize_frame(frame, box, size=16)
    assert resized.shape == (16, 16)


@pytest.fixture
def nifti_file(tmp_path: Path) -> Path:
    volume = _brain_like_volume((32, 32, 20))
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), path)
    return path


def test_load_volume_round_trips_shape(nifti_file: Path) -> None:
    volume = load_volume(nifti_file)
    assert volume.shape == (32, 32, 20)


def test_process_scan_output_shape_and_range(nifti_file: Path) -> None:
    config = ProcessingConfig(n_frames=8, image_size=16)
    result = process_scan(nifti_file, Modality.MRI, config)
    assert result.shape == (16, 16, 8)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_process_scan_default_config_matches_paper_shape(nifti_file: Path) -> None:
    # nifti_file only has depth 20, so use a config within bounds; the
    # paper's reported shape is (128, 128, 50) via `ProcessingConfig()`'s
    # defaults, exercised structurally here with a smaller depth.
    config = ProcessingConfig(n_frames=10, image_size=32)
    result = process_scan(nifti_file, Modality.PET, config)
    assert result.shape == (32, 32, 10)


def _legacy_faithful_reference(
    volume: np.ndarray, config: ProcessingConfig, box_params: BrainBoxParams
) -> np.ndarray:
    """Reimplements the legacy `process_scan`'s order independently of `process_volume`.

    `Dataset_MRI.ipynb`'s `process_scan` calls `normalize(volume)` (full
    volume, own min/max) *then* `resize_to_input_shape` (central-slice,
    brain-crop, resize), with no renormalization afterward. Brain-box
    detection itself (`find_brain_bounding_box`/`_slice_bounding_box`) is
    unaffected by *when* normalization happens: it independently min-max
    rescales each slice to `uint8` for Otsu thresholding, so reusing that
    helper here (rather than duplicating the Otsu/contour pipeline) isolates
    exactly the one variable this test cares about: the order of the global
    normalize step relative to cropping.
    """
    normalized = normalize_intensity(volume)
    central = central_axial_slices(normalized, config.n_frames)
    box = find_brain_bounding_box(central, box_params)
    frames = [
        crop_and_resize_frame(central[..., i], box, config.image_size)
        for i in range(central.shape[-1])
    ]
    return np.stack(frames, axis=-1)


def _buggy_post_crop_reference(
    volume: np.ndarray, config: ProcessingConfig, box_params: BrainBoxParams
) -> np.ndarray:
    """The pre-fix behavior: crop/resize raw values, *then* normalize.

    Normalizes against the crop's own local min/max instead of the full
    scan's, which silently changes every pixel's relative intensity
    whenever the crop excludes the scan's true extrema. Kept here only to
    prove the default order actually differs from this one.
    """
    central = central_axial_slices(volume, config.n_frames)
    box = find_brain_bounding_box(central, box_params)
    frames = [
        crop_and_resize_frame(central[..., i], box, config.image_size)
        for i in range(central.shape[-1])
    ]
    return normalize_intensity(np.stack(frames, axis=-1))


def test_process_scan_normalizes_before_crop_matches_legacy_reference() -> None:
    """Regression test: extrema *outside* the crop must still affect the output.

    Places a huge-value pixel far outside the brain blob (and thus outside
    the detected crop region) but within the sliced depth range. Faithful
    (pre-crop, global) normalization must be dominated by that extremum
    everywhere, including inside the crop; a buggy post-crop normalization
    would never see it and rescale the crop's own (much smaller) local
    range to `[0, 1]` instead, producing a visibly different result.
    """
    shape = (128, 128, 16)
    rng = np.random.default_rng(0)
    volume = rng.normal(50, 5, shape)
    lo, hi = np.array(shape) // 4, np.array(shape) - np.array(shape) // 4
    blob_shape = (hi[0] - lo[0], hi[1] - lo[1], shape[2])
    volume[lo[0] : hi[0], lo[1] : hi[1], :] = rng.normal(500, 50, blob_shape)
    volume = np.clip(volume, 0, None).astype(np.float32)
    # Extremum in a far corner, well outside the blob/crop, present in every
    # sliced depth index so it survives `central_axial_slices` regardless of
    # `n_frames`.
    volume[0, 0, :] = 1.0e5

    config = ProcessingConfig(n_frames=16, image_size=32)
    faithful = _legacy_faithful_reference(volume, config, MRI_BRAIN_BOX_PARAMS)
    buggy = _buggy_post_crop_reference(volume, config, MRI_BRAIN_BOX_PARAMS)

    actual = process_volume(volume, Modality.MRI, config)
    np.testing.assert_array_equal(actual, faithful)
    assert not np.allclose(actual, buggy, atol=1e-2)
    # The buggy order stretches the crop's own local range to fill [0, 1];
    # the faithful order is dominated by the far-away extremum and stays
    # compressed near 0 everywhere inside the crop.
    assert actual.max() < 0.05
    assert buggy.max() > 0.5
