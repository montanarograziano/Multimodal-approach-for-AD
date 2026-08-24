"""Deterministic volume augmentation.

Ports the minority-class augmentation from `Dataset_MRI.ipynb` /
`Dataset_PET.ipynb` (rotate / vertical-flip / horizontal-flip, chained 1-5
times, weighted 0.5/0.25/0.25): see `docs/legacy-notebooks-inventory.md`.

**Documented divergence from the legacy notebook**: augmentation there used
the process-global `random` module (non-reproducible, order-dependent
across cells). This port takes an explicit `numpy.random.Generator`, so
callers control and can reproduce the seed independently of import order or
call order elsewhere in the pipeline. Rotation uses `scipy.ndimage.rotate`
(the modern, non-deprecated location; the legacy
`scipy.ndimage.interpolation.rotate` submodule is gone in current SciPy).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

#: Retained from the legacy notebook: op names, selection weights (must sum
#: to 1), rotation angle range (degrees, half-open), and chained op count
#: range (inclusive).
AUGMENTATION_OPS: tuple[str, ...] = ("rotate", "flipv", "fliph")
AUGMENTATION_WEIGHTS: tuple[float, ...] = (0.5, 0.25, 0.25)
ROTATION_ANGLE_RANGE: tuple[float, float] = (-30.0, 30.0)
OP_COUNT_RANGE: tuple[int, int] = (1, 5)  # inclusive
_CORNER_PATCH_SIZE = 5


def _corner_fill_value(volume: np.ndarray) -> float:
    """Mean intensity of a `(5, 5)` corner patch, used to fill rotated corners."""
    patch = volume[:_CORNER_PATCH_SIZE, :_CORNER_PATCH_SIZE, ...]
    return float(patch.mean())


def _rotate(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(*ROTATION_ANGLE_RANGE)
    fill_value = _corner_fill_value(volume)
    rotated = ndimage_rotate(
        volume,
        angle,
        axes=(0, 1),
        reshape=False,
        mode="constant",
        cval=fill_value,
        order=1,
    )
    return rotated.astype(volume.dtype, copy=False)


def _flip_vertical(volume: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    return np.flip(volume, axis=0)


def _flip_horizontal(volume: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    return np.flip(volume, axis=1)


_OP_FUNCS = {
    "rotate": _rotate,
    "flipv": _flip_vertical,
    "fliph": _flip_horizontal,
}


def augment_volume(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply 1-5 randomly chosen, weighted rotate/flip ops to `volume`.

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
