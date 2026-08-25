"""AAL2 atlas region-importance ranking, ported from `exploration.ipynb`.

**The AAL2 atlas volume is bundled**: `atlas.nii.gz` is checked into the
repository root ((91, 109, 91), 2mm voxels, 120 nonzero region intensities
matching `AAL2_Atlas_Labels.csv`); load it with `load_atlas` below. This is
a research/reference asset only, it is not included in the built sdist/
wheel (see `pyproject.toml`'s sdist allowlist). `AAL2_Atlas_Labels.csv`
(`id, name, intensity` columns, no header) is also checked into the repo
root; load it with `load_region_labels`.

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

**Two mean columns, two different denominators**: the notebook computes
`f"{name} mean"` as `masked.mean()` over the *entire padded common frame*
(`(128, 128, 128)` = 2,097,152 voxels), not over the region alone, because
`masked` is the heatmap zeroed outside the region but still full-frame
shaped (`np.ma.masked_where(...).filled(0)` then `.mean()` with no
per-region slicing). That means `f"{name} mean"` = region sum / *frame*
voxel count, which shrinks toward zero for small regions purely because
the denominator is huge and mostly zeros, not because the region has low
Grad-CAM importance. This module:

- Keeps `f"{name} mean"` computed with the notebook's *exact* formula
  (sum over the region divided by the full common-frame voxel count) as
  the default column, so results reproducing the published paper's
  region tables match the notebook's numbers.
- Adds `f"{name} region mean"`: the *region-count-normalized* mean (sum
  over the region divided by the number of voxels actually in that
  region), i.e. the value most readers would assume "mean Grad-CAM in
  region X" means. This is what should drive new analysis, since it is
  comparable across regions of different sizes; `f"{name} mean"` is not
  (a large region and a tiny region with identical average per-voxel
  signal get very different `f"{name} mean"` values purely from frame-size
  dilution).

`f"{name} sum"` and `f"{name} count"` are unaffected by this distinction
(both are already per-region) and remain as-is; `f"{name} region mean"` is
exactly `sum / count` when `count > 0`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.spatialimages import SpatialImage

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


def load_atlas(path: str | Path = Path("atlas.nii.gz")) -> np.ndarray:
    """Load the bundled AAL2 atlas volume as a `(91, 109, 91)` intensity array.

    Pass the result through `pad_to_frame(atlas, ATLAS_PLACEMENT)` before
    calling `rank_regions` with a same-shaped heatmap.
    """
    # `nib.load` is typed as returning the generic `FileBasedImage` base
    # class, but NIfTI always returns a `SpatialImage` subclass with
    # `get_fdata` (see `data.volumes.load_volume` for the same cast).
    image = cast(SpatialImage, nib.load(path))
    return np.asarray(image.get_fdata())


def load_region_labels(csv_path: str) -> pd.DataFrame:
    """Load `AAL2_Atlas_Labels.csv` (`id, name, intensity` columns, no header).

    Indexed by region `name` (not the leading numeric `id` column), one
    `intensity` column, matching `rank_regions`'s expected `region_labels` shape.
    """
    # `[[...]]` column selection returns `DataFrame | Series` per pandas'
    # stubs; a single-element list always selects a DataFrame at runtime.
    return cast(
        pd.DataFrame,
        pd.read_csv(csv_path, names=["id", "name", "intensity"], index_col="name")[["intensity"]],
    )

    # ponytail: no header-detection/validation beyond pandas defaults; the
    # legacy CSV format is fixed and checked into the repo, add validation
    # if a differently-shaped atlas label file needs supporting.


def rank_regions(
    atlas: np.ndarray,
    region_labels: pd.DataFrame,
    heatmaps: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute masked mean/region-mean/count/sum per atlas region, per heatmap.

    `atlas` and every array in `heatmaps` must already be padded to the same
    shape (e.g. via `pad_to_frame`). `region_labels` is `load_region_labels`'s
    output (index: region name, column `intensity`). Returns one row per
    region with, per heatmap key, mirroring the notebook's per-modality/
    per-class column naming (`"Negative PET mean"`, etc.) without hardcoding
    modality/class names:

    - `f"{name} mean"`: the notebook's exact formula, region sum divided by
      the *full common-frame voxel count* (`atlas.size`), not the region's
      own voxel count. Kept for reproducing the published paper's numbers;
      see module docstring, "Two mean columns, two different denominators".
    - `f"{name} region mean"`: region sum divided by the region's own voxel
      count (`nan` if the region has no matching voxels). The
      size-comparable mean; prefer this for new analysis.
    - `f"{name} count"`: number of nonzero heatmap voxels in the region.
    - `f"{name} sum"`: sum of heatmap values in the region.
    """
    frame_voxel_count = atlas.size
    rows: list[dict[str, object]] = []
    for region_name, row in region_labels.iterrows():
        intensity = row["intensity"]
        region_mask = atlas == intensity
        has_region = bool(region_mask.any())
        record: dict[str, object] = {"part": region_name}
        for heatmap_name, heatmap in heatmaps.items():
            masked = np.where(region_mask, heatmap, 0.0)
            region_sum = float(masked.sum())
            record[f"{heatmap_name} mean"] = region_sum / frame_voxel_count
            record[f"{heatmap_name} region mean"] = (
                float(masked[region_mask].mean()) if has_region else float("nan")
            )
            record[f"{heatmap_name} count"] = int(np.count_nonzero(masked))
            record[f"{heatmap_name} sum"] = region_sum
        rows.append(record)
    return pd.DataFrame(rows)
