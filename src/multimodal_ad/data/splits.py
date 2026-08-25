"""Subject-wise train/test and stratified fold creation, with leakage checks.

**Documented divergence from the legacy notebook** (see
`docs/legacy-notebooks-inventory.md`, `Dataset_MRI.ipynb` step 10 and open
question 5): the notebooks used a fixed positional slice split
(`sample[:75]`, etc.) baked into row-count assumptions, and separately
contained two inconsistent CV fold schemes (subject-level 10x10
`RepeatedStratifiedKFold` vs. frame-level `StratifiedKFold`). Both are
fragile/non-reproducible artifacts, not tuned scientific constants. This
module replaces them with a single canonical, group-aware, seeded scheme:
`sklearn.model_selection.GroupShuffleSplit` for train/test and
`StratifiedGroupKFold` for folds, both keyed on `subject_id` so no subject's
scans can appear in more than one split. Which exact legacy scheme
reproduces the paper's reported numbers remains an open question for the
paper authors; this module's output is not expected to reproduce those
numbers bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

#: Retained from the legacy notebooks (`RANDOM_SEED = 1234`).
DEFAULT_SEED = 1234


class SubjectLeakageError(ValueError):
    """Raised when the same subject appears in more than one split."""


def assert_no_subject_leakage(
    train_df: pl.DataFrame, test_df: pl.DataFrame, *, subject_col: str = "subject_id"
) -> None:
    overlap = set(train_df.get_column(subject_col)) & set(test_df.get_column(subject_col))
    if overlap:
        raise SubjectLeakageError(
            f"subjects present in both splits (no cross-split leakage allowed): {sorted(overlap)}"
        )


def subject_train_test_split(
    df: pl.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    subject_col: str = "subject_id",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split `df` into train/test so that no subject appears in both.

    All rows for a given subject go to the same side of the split.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df.get_column(subject_col)))
    train_df = df[train_idx]
    test_df = df[test_idx]
    assert_no_subject_leakage(train_df, test_df, subject_col=subject_col)
    return train_df, test_df


def stratified_subject_folds(
    df: pl.DataFrame,
    *,
    n_splits: int = 10,
    seed: int = DEFAULT_SEED,
    subject_col: str = "subject_id",
    label_col: str = "label",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build `n_splits` label-stratified folds with subject-level grouping.

    Each fold's `(train_index, test_index)` pair is a row-position array
    into `df` (polars has no separate index; use `df[train_index]` /
    `df[test_index]` to materialize either side). No subject appears in
    both the train and test side of the same fold.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_pos, test_pos in splitter.split(
        df, y=df.get_column(label_col), groups=df.get_column(subject_col)
    ):
        assert_no_subject_leakage(df[train_pos], df[test_pos], subject_col=subject_col)
        folds.append((train_pos, test_pos))
    return folds
