"""Tests for `multimodal_ad.data.splits`."""

import pandas as pd
import pytest

from multimodal_ad.data.splits import (
    SubjectLeakageError,
    assert_no_subject_leakage,
    stratified_subject_folds,
    subject_train_test_split,
)


def _multi_session_df(n_subjects: int = 40, sessions_per_subject: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_subjects):
        label = i % 2
        for s in range(sessions_per_subject):
            rows.append({"subject_id": f"S{i:03d}", "session_id": f"S{i:03d}-{s}", "label": label})
    return pd.DataFrame(rows)


def test_subject_train_test_split_has_no_overlap() -> None:
    df = _multi_session_df()
    train_df, test_df = subject_train_test_split(df, test_size=0.25, seed=1234)

    assert set(train_df["subject_id"]) & set(test_df["subject_id"]) == set()
    assert len(train_df) + len(test_df) == len(df)


def test_subject_train_test_split_keeps_all_sessions_of_a_subject_together() -> None:
    df = _multi_session_df(n_subjects=10, sessions_per_subject=3)
    train_df, test_df = subject_train_test_split(df, test_size=0.3, seed=1234)

    for subject_id, group in df.groupby("subject_id"):
        in_train = subject_id in set(train_df["subject_id"])
        in_test = subject_id in set(test_df["subject_id"])
        assert in_train != in_test  # exactly one side, never both, never neither
        assert len(group) == len(train_df[train_df["subject_id"] == subject_id]) + len(
            test_df[test_df["subject_id"] == subject_id]
        )


def test_subject_train_test_split_is_deterministic() -> None:
    df = _multi_session_df()
    train_a, test_a = subject_train_test_split(df, seed=1234)
    train_b, test_b = subject_train_test_split(df, seed=1234)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_stratified_subject_folds_have_no_leakage() -> None:
    df = _multi_session_df(n_subjects=40, sessions_per_subject=2)
    folds = stratified_subject_folds(df, n_splits=5, seed=1234)

    assert len(folds) == 5
    for train_index, test_index in folds:
        train_subjects = set(df.loc[train_index, "subject_id"])
        test_subjects = set(df.loc[test_index, "subject_id"])
        assert train_subjects & test_subjects == set()


def test_assert_no_subject_leakage_raises_on_overlap() -> None:
    train_df = pd.DataFrame({"subject_id": ["A", "B"]})
    test_df = pd.DataFrame({"subject_id": ["B", "C"]})
    with pytest.raises(SubjectLeakageError, match="B"):
        assert_no_subject_leakage(train_df, test_df)


def test_assert_no_subject_leakage_passes_on_disjoint_sets() -> None:
    train_df = pd.DataFrame({"subject_id": ["A", "B"]})
    test_df = pd.DataFrame({"subject_id": ["C", "D"]})
    assert_no_subject_leakage(train_df, test_df)  # must not raise
