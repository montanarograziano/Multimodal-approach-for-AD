"""Tests for `multimodal_ad.data.splits`."""

import polars as pl
import pytest

from multimodal_ad.data.splits import (
    SubjectLeakageError,
    assert_no_subject_leakage,
    stratified_subject_folds,
    subject_train_test_split,
)


def _multi_session_df(n_subjects: int = 40, sessions_per_subject: int = 3) -> pl.DataFrame:
    rows = []
    for i in range(n_subjects):
        label = i % 2
        for s in range(sessions_per_subject):
            rows.append({"subject_id": f"S{i:03d}", "session_id": f"S{i:03d}-{s}", "label": label})
    return pl.DataFrame(rows)


def test_subject_train_test_split_has_no_overlap() -> None:
    df = _multi_session_df()
    train_df, test_df = subject_train_test_split(df, test_size=0.25, seed=1234)

    assert set(train_df.get_column("subject_id")) & set(test_df.get_column("subject_id")) == set()
    assert len(train_df) + len(test_df) == len(df)


def test_subject_train_test_split_keeps_all_sessions_of_a_subject_together() -> None:
    df = _multi_session_df(n_subjects=10, sessions_per_subject=3)
    train_df, test_df = subject_train_test_split(df, test_size=0.3, seed=1234)

    train_subjects = train_df.get_column("subject_id")
    test_subjects = test_df.get_column("subject_id")
    for (subject_id,), group in df.group_by("subject_id"):
        in_train = subject_id in set(train_subjects)
        in_test = subject_id in set(test_subjects)
        assert in_train != in_test  # exactly one side, never both, never neither
        n_train = train_df.filter(pl.col("subject_id") == subject_id).height
        n_test = test_df.filter(pl.col("subject_id") == subject_id).height
        assert len(group) == n_train + n_test


def test_subject_train_test_split_is_deterministic() -> None:
    df = _multi_session_df()
    train_a, test_a = subject_train_test_split(df, seed=1234)
    train_b, test_b = subject_train_test_split(df, seed=1234)

    assert train_a.equals(train_b)
    assert test_a.equals(test_b)


def test_stratified_subject_folds_have_no_leakage() -> None:
    df = _multi_session_df(n_subjects=40, sessions_per_subject=2)
    folds = stratified_subject_folds(df, n_splits=5, seed=1234)

    assert len(folds) == 5
    for train_index, test_index in folds:
        train_subjects = set(df[train_index].get_column("subject_id"))
        test_subjects = set(df[test_index].get_column("subject_id"))
        assert train_subjects & test_subjects == set()


def test_assert_no_subject_leakage_raises_on_overlap() -> None:
    train_df = pl.DataFrame({"subject_id": ["A", "B"]})
    test_df = pl.DataFrame({"subject_id": ["B", "C"]})
    with pytest.raises(SubjectLeakageError, match="B"):
        assert_no_subject_leakage(train_df, test_df)


def test_assert_no_subject_leakage_passes_on_disjoint_sets() -> None:
    train_df = pl.DataFrame({"subject_id": ["A", "B"]})
    test_df = pl.DataFrame({"subject_id": ["C", "D"]})
    assert_no_subject_leakage(train_df, test_df)  # must not raise
