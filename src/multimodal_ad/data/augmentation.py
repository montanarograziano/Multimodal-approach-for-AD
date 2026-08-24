"""Deterministic volume augmentation on raw, full-resolution volumes.

Ports `transform`/`rotate_img` from `Dataset_MRI.ipynb` / `Dataset_PET.ipynb`
(rotate / vertical-flip / horizontal-flip, chained 1-5 times, weighted
0.5/0.25/0.25): see `docs/legacy-notebooks-inventory.md`.

**Contract**: `augment_volume` operates on *raw* volumes as loaded from disk
(e.g. `load_volume`/`average_4d_frames` output), exactly like the legacy
`transform()`, which ran on `nii_img.dataobj` before any
normalize/slice/crop/resize step. Run `data.volumes.process_volume` on the
result afterward to get a model-ready array; do not augment already-processed
volumes.

Ported verbatim from the legacy notebook:
- Rotation loops over `volume[i, :, :]` (axis 0, matching the legacy loop
  exactly; this is *not* the axial/depth axis), applying one shared random
  integer angle per `rotate` op to every slice.
- The rotation angle is an integer in `[-30, 30)`, matching
  `random.randrange(-30, 30)` (`np.random.Generator.integers` defaults to
  the same half-open, integer-only range).
- Each rotated slice's exposed corners/background (any pixel `<= 0` after
  rotation) are refilled with that *same slice's* pre-rotation `(5, 5)`
  top-left corner mean, matching the legacy per-slice `bg_color`/mask.
- Flip axes are the legacy `transformed[:, ::-1, :]` (`flipv`) and
  `transformed[:, :, ::-1]` (`fliph`); axis 0 is never flipped.

**Documented divergences from the legacy notebook**:
- The legacy notebook used the process-global `random` module
  (non-reproducible, import/call-order-dependent). This port takes an
  explicit `numpy.random.Generator`, so callers control and can reproduce
  the seed independently.
- Rotation uses `scipy.ndimage.rotate` (the modern, non-deprecated
  location; the legacy `scipy.ndimage.interpolation.rotate` submodule is
  gone in current SciPy) called with the same defaults the legacy code
  relied on implicitly (`reshape=False`, cubic-spline `order=3`,
  zero-constant `mode="constant"`). Spline interpolation coefficients can
  differ in the last few ULPs across SciPy versions; this is an unavoidable
  numeric difference, not a behavioral one.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

#: Retained from the legacy notebook: op names, selection weights (must sum
#: to 1), and chained op count range (inclusive).
AUGMENTATION_OPS: tuple[str, ...] = ("rotate", "flipv", "fliph")
AUGMENTATION_WEIGHTS: tuple[float, ...] = (0.5, 0.25, 0.25)
#: Half-open integer range, matching legacy `random.randrange(-30, 30)`.
ROTATION_ANGLE_RANGE: tuple[int, int] = (-30, 30)
OP_COUNT_RANGE: tuple[int, int] = (1, 5)  # inclusive
_CORNER_PATCH_SIZE = 5


def _corner_background(frame: np.ndarray) -> float:
    """Mean intensity of a 2D frame's `(5, 5)` top-left corner (pre-rotation)."""
    return float(frame[:_CORNER_PATCH_SIZE, :_CORNER_PATCH_SIZE].mean())


def _rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
    """Port of `rotate_img` for a single 2D slice."""
    background = _corner_background(frame)
    rotated = ndimage_rotate(frame, angle, reshape=False)
    rotated[rotated <= 0] = background
    return rotated.astype(frame.dtype, copy=False)


def _rotate(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One shared random angle, applied to every `volume[i, :, :]` slice."""
    angle = int(rng.integers(*ROTATION_ANGLE_RANGE))
    out = volume.copy()
    for i in range(volume.shape[0]):
        out[i, :, :] = _rotate_frame(volume[i, :, :], angle)
    return out


def _flip_vertical(volume: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    """`transformed[:, ::-1, :]` in the legacy notebook."""
    return volume[:, ::-1, :].copy()


def _flip_horizontal(volume: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    """`transformed[:, :, ::-1]` in the legacy notebook."""
    return volume[:, :, ::-1].copy()


_OP_FUNCS = {
    "rotate": _rotate,
    "flipv": _flip_vertical,
    "fliph": _flip_horizontal,
}


def augment_volume(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply 1-5 randomly chosen, weighted rotate/flip ops to a raw volume.

    See module docstring: `volume` must be raw/unprocessed (pre-`process_volume`).
    Deterministic given `rng`'s seed. Intended for minority-class
    oversampling; call only on the training split (see `data.splits`) to
    avoid leaking augmented copies of the same source volume across splits.
    """
    n_ops = int(rng.integers(OP_COUNT_RANGE[0], OP_COUNT_RANGE[1] + 1))
    out = volume
    for _ in range(n_ops):
        op = rng.choice(AUGMENTATION_OPS, p=AUGMENTATION_WEIGHTS)
        out = _OP_FUNCS[str(op)](out, rng)
    return out
