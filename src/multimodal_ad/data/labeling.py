"""Diagnosis normalization and longitudinal (temporal) label correction.

Ports the labeling logic from `Dataset_MRI.ipynb` / `Dataset_PET.ipynb`
(see `docs/legacy-notebooks-inventory.md`), preserving its scientifically
meaningful decisions:

- The demented/not-demented regex classification, including treating
  "uncertain" diagnoses as demented (the notebooks' comment: "we classify
  uncertain cases as sick").
- The temporal smoothing intent: an isolated non-demented visit, when the
  nearby visits (within a 2-visit window on both sides) are demented, is
  corrected to demented, to counter visit-level diagnostic noise in a
  disease that is expected to be progressive/monotonic in this cohort.

**Documented divergence from the legacy notebook** (see inventory
Ambiguity notes for `Dataset_MRI.ipynb` step 3): the original loop's
behavior at the first/last 1-2 rows per subject was underspecified
(off-by-one-prone manual indexing, no tests). This port defines an explicit,
deterministic rule instead of guessing at the original edge behavior: a
`False` visit is corrected only if *both* a demented visit exists within
the 2 preceding visits *and* a demented visit exists within the 2 following
visits (a boundary visit with fewer than 2 neighbors on one side can never
satisfy this and is left uncorrected). This must be confirmed against the
paper's reported numbers before being treated as final (see
`docs/legacy-notebooks-inventory.md`, open question 4).

The garbled `other mental retarAD demion` diagnosis fragment from the
legacy regex is dropped here as unrecoverable noise rather than guessed at;
if the OASIS-3 codebook clarifies its intent, this regex should be updated.
"""

from __future__ import annotations

import re

import pandas as pd

# Case-insensitive; matches the legacy regex minus the garbled fragment
# `other mental retarAD demion` (unrecoverable, see module docstring).
_DEMENTED_PATTERN = re.compile(
    r"^(AD dem|Vasc.*? dem|Frontotemporal dem|(Active )?DLBD|Active PSNP|Dementia)",
    re.IGNORECASE,
)
_UNCERTAIN_PATTERN = re.compile(r"^uncertain.*?dem", re.IGNORECASE)

#: Number of preceding/following visits considered when smoothing an
#: isolated non-demented reading (retained from the legacy notebook).
TEMPORAL_SMOOTHING_WINDOW = 2


def classify_diagnosis(diagnosis: str | float | None) -> bool:
    """Classify a single free-text differential-diagnosis string.

    Returns `True` for demented/uncertain-demented diagnoses, `False`
    otherwise (including missing/NaN values).
    """
    if diagnosis is None or (isinstance(diagnosis, float) and pd.isna(diagnosis)):
        return False
    text = str(diagnosis)
    return bool(_DEMENTED_PATTERN.match(text) or _UNCERTAIN_PATTERN.match(text))


def classify_diagnosis_row(dx_values: pd.Series | list[str | float | None]) -> bool:
    """Classify a row with multiple differential-diagnosis columns (dx1..dxN).

    A row is demented if *any* of its diagnosis fields classify as demented.
    """
    return any(classify_diagnosis(dx) for dx in dx_values)


def smooth_temporal_labels(
    labels: pd.Series,
    subject_ids: pd.Series,
    day_offsets: pd.Series,
    *,
    window: int = TEMPORAL_SMOOTHING_WINDOW,
) -> pd.Series:
    """Correct isolated non-demented readings surrounded by demented visits.

    `labels`, `subject_ids`, and `day_offsets` must be aligned (same index,
    same length). Rows are ordered by `day_offset` within each subject
    before smoothing, and the result is returned reindexed to match the
    input's original index/order.
    """
    if not (len(labels) == len(subject_ids) == len(day_offsets)):
        raise ValueError("labels, subject_ids, and day_offsets must have the same length")

    frame = pd.DataFrame(
        {
            "label": labels.to_numpy(),
            "subject_id": subject_ids.to_numpy(),
            "day": day_offsets.to_numpy(),
        },
        index=labels.index,
    )
    smoothed = frame["label"].copy()

    for _subject, group in frame.groupby("subject_id", sort=False):
        ordered = group.sort_values("day")
        values = ordered["label"].tolist()
        corrected = list(values)
        for i, value in enumerate(values):
            if value:
                continue
            has_prior_true = any(values[max(0, i - window) : i])
            has_next_true = any(values[i + 1 : i + 1 + window])
            if has_prior_true and has_next_true:
                corrected[i] = True
        smoothed.loc[ordered.index] = corrected

    return smoothed.reindex(labels.index)


def label_nearest_visit(
    scan_subject_ids: pd.Series,
    scan_day_offsets: pd.Series,
    clinical: pd.DataFrame,
    *,
    clinical_subject_col: str = "subject_id",
    clinical_day_col: str = "day_offset",
    clinical_label_col: str = "label",
) -> pd.Series:
    """Assign each scan the label of its nearest-in-time clinical visit.

    For each scan, finds the clinical visit for the same subject minimizing
    `abs(day_offset - visit_day)` and returns that visit's label. Ties are
    broken by the first matching row in `clinical`'s given order (documented
    here; the legacy notebook relied on undocumented pandas sort stability).
    Scans for a subject with no clinical visits get `pd.NA`.
    """
    if len(scan_subject_ids) != len(scan_day_offsets):
        raise ValueError("scan_subject_ids and scan_day_offsets must have the same length")

    grouped_clinical = {
        subject: group for subject, group in clinical.groupby(clinical_subject_col, sort=False)
    }

    results: list[object] = []
    for subject, day in zip(scan_subject_ids, scan_day_offsets, strict=True):
        visits = grouped_clinical.get(subject)
        if visits is None or visits.empty:
            results.append(pd.NA)
            continue
        deltas = (visits[clinical_day_col] - day).abs()
        nearest_index = deltas.idxmin()
        results.append(visits.loc[nearest_index, clinical_label_col])

    return pd.Series(results, index=scan_subject_ids.index)
