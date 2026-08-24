"""MRI/PET volume loading, frame handling, brain cropping, and resizing.

Ports `process_scan` / `find_brain_bounding_box` / `resize_to_input_shape`
from `Dataset_MRI.ipynb` / `Dataset_PET.ipynb` (see
`docs/legacy-notebooks-inventory.md`), preserving the modality-specific
Otsu tuning as an explicit, named preset rather than a copy-paste
coincidence.

**Documented divergence from the legacy notebook**: `find_brain_bounding_box`
originally called `cv2.findContours(...)[0]` unconditionally, raising
`IndexError` on a slice with no contour (flagged as a bug, not a scientific
choice, in the inventory). This port falls back to the full-frame bounding
box for such slices instead of raising, so a single degenerate slice does
not crash the whole volume's processing.

**Preprocessing order (fidelity fix)**: the legacy `process_scan` calls
`normalize(volume)` on the *full* loaded/frame-averaged volume, then
`resize_to_input_shape(volume)` (central-slice, brain-crop, resize) on the
already-normalized data, with no renormalization afterward. `process_volume`
replicates that order exactly: normalize first, using the full volume's own
min/max, then slice/crop/resize. An earlier version of this port normalized
*after* cropping, which rescales each scan against its post-crop min/max
instead of the whole scan's, silently changing every pixel's relative
intensity whenever the crop excludes the volume's true extrema (see
`tests/data/test_volumes.py::test_process_scan_normalizes_before_crop_matches_legacy_reference`
for a regression case that distinguishes the two orders). There is
deliberately no "optimized" post-crop-normalize preset: nothing in this
codebase needs one, and adding a config knob for an already-known-wrong
behavior would just be a footgun.

**Open ambiguity** (see inventory, open question 1): the legacy notebooks
process 20, 30, or 50 central frames in different cells, and it is not
determinable from the notebooks alone which produced the paper's reported
results. This port defaults `n_frames=50` (the paper's reported input
depth, `(128, 128, 50)`) and exposes it as a `ProcessingConfig` field so
callers can reproduce the other experimental variants explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

import cv2
import nibabel as nib
import numpy as np

from multimodal_ad.data.manifest import Modality


class BoundingBox(NamedTuple):
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class BrainBoxParams:
    """Otsu-thresholding parameters for `find_brain_bounding_box`."""

    gaussian_ksize: tuple[int, int]
    gaussian_sigma: float
    median_ksize: int
    padding: int = 10


#: Modality-specific Otsu tuning retained verbatim from the legacy notebooks.
MRI_BRAIN_BOX_PARAMS = BrainBoxParams(gaussian_ksize=(51, 51), gaussian_sigma=50, median_ksize=51)
PET_BRAIN_BOX_PARAMS = BrainBoxParams(gaussian_ksize=(13, 13), gaussian_sigma=150, median_ksize=5)

_DEFAULT_BOX_PARAMS: dict[Modality, BrainBoxParams] = {
    Modality.MRI: MRI_BRAIN_BOX_PARAMS,
    Modality.PET: PET_BRAIN_BOX_PARAMS,
}


def default_box_params(modality: Modality) -> BrainBoxParams:
    return _DEFAULT_BOX_PARAMS[modality]


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Configuration for `process_scan`. See module docstring for `n_frames`."""

    n_frames: int = 50
    image_size: int = 128
    box_params: BrainBoxParams | None = None  # None => modality default


def load_volume(path: Path) -> np.ndarray:
    """Load a scan (NIfTI `.nii`/`.nii.gz` or Analyze `.img`/`.hdr`) as an ndarray."""
    # `nib.load` is typed as returning the generic `FileBasedImage` base class,
    # but every format this pipeline supports (NIfTI, Analyze) returns a
    # `SpatialImage` subclass with `get_fdata`.
    image = cast(nib.spatialimages.SpatialImage, nib.load(path))
    return np.asarray(image.get_fdata())


def average_4d_frames(volume: np.ndarray) -> np.ndarray:
    """Average a 4D (x, y, z, frame) volume over its frame axis to 3D.

    Some MRI/PET files carry a 4th "echo"/frame axis; the legacy pipeline
    collapses it via a plain mean. 3D input is passed through unchanged
    (cast to float32).
    """
    if volume.ndim == 4:
        return volume.mean(axis=3, dtype=np.float32)
    if volume.ndim == 3:
        return volume.astype(np.float32, copy=False)
    raise ValueError(f"expected a 3D or 4D volume, got ndim={volume.ndim}")


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Min-max scale a volume to `[0, 1]` float32, using its own min/max."""
    volume = volume.astype(np.float32, copy=False)
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - vmin) / (vmax - vmin)


def central_axial_slices(volume: np.ndarray, n_frames: int) -> np.ndarray:
    """Extract the `n_frames` central slices along the last (axial/depth) axis."""
    depth = volume.shape[-1]
    if n_frames > depth:
        raise ValueError(f"n_frames={n_frames} exceeds volume depth={depth}")
    start = (depth - n_frames) // 2
    return volume[..., start : start + n_frames]


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    vmin, vmax = float(frame.min()), float(frame.max())
    if vmax <= vmin:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = (frame - vmin) / (vmax - vmin) * 255.0
    return scaled.astype(np.uint8)


def _slice_bounding_box(frame: np.ndarray, params: BrainBoxParams) -> BoundingBox:
    """Otsu-threshold a single 2D frame and return its largest contour's bbox.

    Falls back to the full-frame bbox when no contour is found (see module
    docstring for the divergence from the legacy notebook's unguarded
    `cv2.findContours(...)[0]`).
    """
    height, width = frame.shape
    gray = _to_uint8(frame)
    blurred = cv2.GaussianBlur(gray, params.gaussian_ksize, params.gaussian_sigma)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresholded, params.median_ksize)
    contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BoundingBox(0, 0, width, height)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return BoundingBox(x, y, w, h)


def find_brain_bounding_box(volume: np.ndarray, params: BrainBoxParams) -> BoundingBox:
    """Union the per-slice brain bounding box across all slices, then pad.

    `volume` should already be restricted to the slices of interest (e.g.
    via `central_axial_slices`); the legacy notebook unions across its 50
    central slices specifically.
    """
    height, width = volume.shape[0], volume.shape[1]
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    for i in range(volume.shape[-1]):
        box = _slice_bounding_box(volume[..., i], params)
        x_min = min(x_min, box.x)
        y_min = min(y_min, box.y)
        x_max = max(x_max, box.x + box.width)
        y_max = max(y_max, box.y + box.height)

    x_min = max(0, x_min - params.padding)
    y_min = max(0, y_min - params.padding)
    x_max = min(width, x_max + params.padding)
    y_max = min(height, y_max + params.padding)
    return BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)


def crop_and_resize_frame(frame: np.ndarray, box: BoundingBox, size: int) -> np.ndarray:
    """Square-crop `frame` around `box` (padding the shorter side) and resize to `size`x`size`."""
    height, width = frame.shape
    side = max(box.width, box.height)
    cx, cy = box.x + box.width // 2, box.y + box.height // 2
    half = side // 2

    x0, x1 = cx - half, cx - half + side
    y0, y1 = cy - half, cy - half + side
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)

    cropped = frame[y0:y1, x0:x1]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)


def process_volume(
    volume: np.ndarray, modality: Modality, config: ProcessingConfig | None = None
) -> np.ndarray:
    """Normalize, center-slice, brain-crop, and resize an already-loaded volume.

    `volume` should be a raw (or 4D-frame-averaged) volume, e.g. straight out
    of `load_volume`/`average_4d_frames`, or a `data.augmentation.augment_volume`
    output (augmentation runs on raw volumes *before* this function; see that
    module). Returns a float32 array of shape `(image_size, image_size,
    n_frames)` with values in `[0, 1]`, normalized against the full input
    volume's own min/max (see module docstring: "Preprocessing order").
    """
    config = config or ProcessingConfig()
    box_params = config.box_params or default_box_params(modality)

    normalized = normalize_intensity(volume)
    central = central_axial_slices(normalized, config.n_frames)
    box = find_brain_bounding_box(central, box_params)

    frames = [
        crop_and_resize_frame(central[..., i], box, config.image_size)
        for i in range(central.shape[-1])
    ]
    return np.stack(frames, axis=-1)


def process_scan(
    path: Path, modality: Modality, config: ProcessingConfig | None = None
) -> np.ndarray:
    """Load a scan file and run it through `process_volume`.

    Returns a float32 array of shape `(image_size, image_size, n_frames)`
    with values in `[0, 1]`.
    """
    volume = load_volume(path)
    volume = average_4d_frames(volume)
    return process_volume(volume, modality, config)
