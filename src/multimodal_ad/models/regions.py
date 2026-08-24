"""AAL2 atlas region-importance ranking, ported from `exploration.ipynb`.

**Requires the AAL2 atlas volume separately**: `atlas.nii.gz` is not checked
into this repository (see inventory doc, open question 6); callers must
source it (e.g. from NeuroVault, as referenced in the notebook's markdown)
and load it with `nibabel` before calling this module. `AAL2_Atlas_Labels.csv`
(region name -> intensity value, no header) *is* checked into the repo root.

**Retained, spatial padding**: the notebook pads both the `(128, 128, 50)`
Grad-CAM heatmap and the atlas volume into a common `(128, 128, 128)` frame
using hardcoded offset slices (`fix_heat_dim`/`fix_atlas_dim`): heatmap at
`[:, :, 39:89]`, atlas at `[19:110, 10:119, 19:110]`. **Documented
limitation** (per inventory doc): these offsets assume an unverified
spatial alignment between the AAL2 atlas's native resolution/origin and the
pipeline's resized MRI/PET volumes; there is no registration step in the
notebook. This module retains the offsets as-is (they are not
rederivable/improvable without a real registration step, which is out of
scope here) and documents them as a known limitation, not a validated
coregistration. `pad_to_frame` below generalizes the two hardcoded
functions into one, parameterized by the placement slice.

**Retained, ranking semantics**: masked mean/count/sum per atlas region,
per `{modality} x {class}` heatmap. The notebook sorts by ascending mean
value and calls `.head()` the "best" regions, implicitly treating *lower*
Grad-CAM values as more important (a jet-colormap reading convention never
confirmed in the notebook's comments). This module exposes the raw
per-region means (`rank_regions`) without built-in ascending/descending
"best" semantics; callers must choose and document a sort direction
explicitly rather than inherit the notebook's unconfirmed convention (see
inventory doc, `exploration.ipynb` section, final bullet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Retained from the legacy notebook's `fix_heat_dim`/`fix_atlas_dim`.
COMMON_FRAME_SHAPE = (128, 128, 128)
HEATMAP_PLACEMENT = (slice(None), slice(None), slice(39, 89))
ATLAS_PLACEMENT = (slice(19, 110), slice(10, 119), slice(19, 110))


def pad_to_frame(
    volume: np.ndarray,
    placement: tuple[slice, slice, slice],
    *,
    frame_shape: tuple[int, int, int] = COMMON_FRAME_SHAPE,
) -> np.ndarray:
    """Place `volume` into a zero-filled `frame_shape` volume at `placement`.

    Raises `ValueError` if `volume`'s shape does not match the extent
    implied by `placement` (the notebook had no such check and would raise
    an opaque `ValueError` from NumPy broadcasting on a shape mismatch).
    """
    frame = np.zeros(frame_shape, dtype=np.float64)
    target_shape = frame[placement].shape
    if volume.shape != target_shape:
        raise ValueError(
            f"volume shape {volume.shape} does not match placement extent {target_shape}"
        )
    frame[placement] = volume
    return frame


def load_region_labels(csv_path: str) -> pd.DataFrame:
    """Load `AAL2_Atlas_Labels.csv` (`name, intensity` columns, no header)."""
    return pd.read_csv(csv_path, names=["name", "intensity"], index_col=0)

    # ponytail: no header-detection/validation beyond pandas defaults; the
    # legacy CSV format is fixed and checked into the repo, add validation
    # if a differently-shaped atlas label file needs supporting.


def rank_regions(
    atlas: np.ndarray,
    region_labels: pd.DataFrame,
    heatmaps: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute masked mean/count/sum per atlas region, for each named heatmap.

    `atlas` and every array in `heatmaps` must already be padded to the same
    shape (e.g. via `pad_to_frame`). `region_labels` is `load_region_labels`'s
    output (index: region name, column `intensity`). Returns one row per
    region with `f"{name} mean"`, `f"{name} count"`, `f"{name} sum"` columns
    per heatmap key, mirroring the notebook's per-modality/per-class column
    naming (`"Negative PET mean"`, etc.) without hardcoding modality/class
    names.
    """
    rows: list[dict[str, object]] = []
    for region_name, row in region_labels.iterrows():
        intensity = row["intensity"]
        region_mask = atlas == intensity
        record: dict[str, object] = {"part": region_name}
        for heatmap_name, heatmap in heatmaps.items():
            masked = np.where(region_mask, heatmap, 0.0)
            has_region = bool(region_mask.any())
            record[f"{heatmap_name} mean"] = (
                float(masked[region_mask].mean()) if has_region else float("nan")
            )
            record[f"{heatmap_name} count"] = int(np.count_nonzero(masked))
            record[f"{heatmap_name} sum"] = float(masked.sum())
        rows.append(record)
    return pd.DataFrame(rows)
