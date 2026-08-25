"""Tests for `multimodal_ad.data.augmentation`."""

import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

from multimodal_ad.data.augmentation import (
    _CORNER_PATCH_SIZE,
    _flip_horizontal,
    _flip_vertical,
    _rotate_frame,
    augment_volume,
)


def test_augment_volume_preserves_shape() -> None:
    rng = np.random.default_rng(0)
    volume = np.random.default_rng(1).uniform(0, 1, (16, 16, 8)).astype(np.float32)
    augmented = augment_volume(volume, rng)
    assert augmented.shape == volume.shape
    assert augmented.dtype == volume.dtype
    # Deliberately *not* asserting a tight [min, max] bound here: the legacy
    # `rotate_img` uses `scipy.ndimage.rotate`'s default cubic-spline
    # interpolation (`order=3`), which can genuinely overshoot the input
    # range at edges (Gibbs-like ringing). Matching that exactly is the
    # point of this port; see `test_rotate_frame_matches_legacy_reference`.


def test_augment_volume_is_deterministic_given_seed() -> None:
    volume = np.random.default_rng(1).uniform(0, 1, (16, 16, 8)).astype(np.float32)

    result_a = augment_volume(volume, np.random.default_rng(42))
    result_b = augment_volume(volume, np.random.default_rng(42))

    np.testing.assert_array_equal(result_a, result_b)


def test_augment_volume_different_seeds_usually_differ() -> None:
    volume = np.random.default_rng(1).uniform(0, 1, (16, 16, 8)).astype(np.float32)

    result_a = augment_volume(volume, np.random.default_rng(1))
    result_b = augment_volume(volume, np.random.default_rng(2))

    assert not np.array_equal(result_a, result_b)


def test_flip_vertical_matches_legacy_reference() -> None:
    """Legacy `transform()`: `flipv` -> `transformed[:, ::-1, :]` (axis 1)."""
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    result = _flip_vertical(volume, np.random.default_rng(0))
    reference = volume[:, ::-1, :]
    np.testing.assert_array_equal(result, reference)
    # Axis 0 (the legacy rotation loop axis) must be untouched by flips.
    np.testing.assert_array_equal(result[:, ::-1, :], volume)


def test_flip_horizontal_matches_legacy_reference() -> None:
    """Legacy `transform()`: `fliph` -> `transformed[:, :, ::-1]` (axis 2)."""
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    result = _flip_horizontal(volume, np.random.default_rng(0))
    reference = volume[:, :, ::-1]
    np.testing.assert_array_equal(result, reference)


def _legacy_rotate_img(
    img: np.ndarray, angle: int, bg_patch: tuple[int, int] = (5, 5)
) -> np.ndarray:
    """Independent reimplementation of the legacy notebook's `rotate_img` (grayscale path).

    Transcribed directly from `Dataset_MRI.ipynb` (see task context), not by
    calling into `multimodal_ad.data.augmentation`, so this is a true
    reference rather than a self-contract check.
    """
    bg_color = np.mean(img[: bg_patch[0], : bg_patch[1]])
    rotated = ndimage_rotate(img, angle, reshape=False)
    mask = rotated <= 0
    rotated[mask] = bg_color
    return rotated


def test_rotate_frame_matches_legacy_reference() -> None:
    frame = np.random.default_rng(3).uniform(0.1, 1.0, (32, 32)).astype(np.float32)
    for angle in (-30, -1, 0, 7, 29):
        expected = _legacy_rotate_img(frame.copy(), angle)
        actual = _rotate_frame(frame.copy(), angle)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_rotate_frame_fills_corners_with_pre_rotation_corner_mean() -> None:
    # A 90-degree rotation exposes no new background (it's an exact pixel
    # permutation), so instead force exposed corners with a large angle and
    # confirm the fill value is that slice's own pre-rotation corner mean,
    # not e.g. a whole-volume mean or zero.
    frame = np.full((32, 32), 1.0, dtype=np.float32)
    frame[:_CORNER_PATCH_SIZE, :_CORNER_PATCH_SIZE] = 9.0
    rotated = _rotate_frame(frame.copy(), 45)
    expected_bg = 9.0
    assert np.any(rotated == expected_bg)
