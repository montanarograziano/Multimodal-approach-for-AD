"""Tests for `multimodal_ad.data.labeling`."""

import polars as pl

from multimodal_ad.data.labeling import (
    classify_diagnosis,
    classify_diagnosis_row,
    label_nearest_visit,
    smooth_temporal_labels,
)


def test_classify_diagnosis_positive_cases() -> None:
    assert classify_diagnosis("AD dem/Vascular") is True
    assert classify_diagnosis("Vasc Dem, sudden onset") is True
    assert classify_diagnosis("Frontotemporal demt., primary") is True
    assert classify_diagnosis("Active DLBD") is True
    assert classify_diagnosis("DLBD") is True
    assert classify_diagnosis("Active PSNP") is True
    assert classify_diagnosis("Dementia, NOS") is True
    assert classify_diagnosis("uncertain dementia") is True
    assert classify_diagnosis("Uncertain, possible dem.") is True


def test_classify_diagnosis_negative_cases() -> None:
    assert classify_diagnosis("Cognitively normal") is False
    assert classify_diagnosis("MCI") is False
    assert classify_diagnosis(None) is False
    assert classify_diagnosis(float("nan")) is False
    assert classify_diagnosis("") is False


def test_classify_diagnosis_row_any_column_matches() -> None:
    assert classify_diagnosis_row(["Cognitively normal", None, "AD dem"]) is True
    assert classify_diagnosis_row(["Cognitively normal", None, None]) is False


def test_smooth_temporal_labels_corrects_isolated_dip() -> None:
    # Subject A: True, False, True (isolated dip surrounded by True on both sides).
    labels = pl.Series([True, False, True])
    subjects = pl.Series(["A", "A", "A"])
    days = pl.Series([0, 30, 60])

    smoothed = smooth_temporal_labels(labels, subjects, days)
    assert smoothed.to_list() == [True, True, True]


def test_smooth_temporal_labels_leaves_unsurrounded_false() -> None:
    # False at the start has no preceding True: must not be corrected
    # (documented, deliberate edge-case resolution; see module docstring).
    labels = pl.Series([False, True, True])
    subjects = pl.Series(["A", "A", "A"])
    days = pl.Series([0, 30, 60])

    smoothed = smooth_temporal_labels(labels, subjects, days)
    assert smoothed.to_list() == [False, True, True]


def test_smooth_temporal_labels_respects_window_size() -> None:
    # Three consecutive False values exceed the 2-visit window: only the
    # ones within range of a True on both sides are corrected.
    labels = pl.Series([True, False, False, False, True])
    subjects = pl.Series(["A"] * 5)
    days = pl.Series([0, 10, 20, 30, 40])

    smoothed = smooth_temporal_labels(labels, subjects, days)
    # Only index 2 has a True within 2 steps on *both* sides (index 0 back,
    # index 4 forward); indices 1 and 3 each miss a True on one side.
    assert smoothed.to_list() == [True, False, True, False, True]


def test_smooth_temporal_labels_is_per_subject() -> None:
    labels = pl.Series([True, False, True, False, False, False])
    subjects = pl.Series(["A", "A", "A", "B", "B", "B"])
    days = pl.Series([0, 10, 20, 0, 10, 20])

    smoothed = smooth_temporal_labels(labels, subjects, days)
    assert smoothed.to_list() == [True, True, True, False, False, False]


def test_smooth_temporal_labels_out_of_order_input_is_sorted_by_day() -> None:
    labels = pl.Series([True, True, False])
    subjects = pl.Series(["A", "A", "A"])
    days = pl.Series([20, 0, 10])  # unordered: day-order is True(0), False(10), True(20)

    smoothed = smooth_temporal_labels(labels, subjects, days)
    assert smoothed.to_list() == [True, True, True]


def test_label_nearest_visit_picks_closest_day() -> None:
    clinical = pl.DataFrame(
        {
            "subject_id": ["A", "A", "A"],
            "day_offset": [0, 50, 100],
            "label": [0, 1, 0],
        }
    )
    scan_subjects = pl.Series(["A", "A"])
    scan_days = pl.Series([45, 99])

    labels = label_nearest_visit(scan_subjects, scan_days, clinical)
    assert labels.to_list() == [1, 0]


def test_label_nearest_visit_missing_subject_is_na() -> None:
    clinical = pl.DataFrame({"subject_id": ["A"], "day_offset": [0], "label": [1]})
    scan_subjects = pl.Series(["B"])
    scan_days = pl.Series([0])

    labels = label_nearest_visit(scan_subjects, scan_days, clinical)
    assert labels[0] is None
