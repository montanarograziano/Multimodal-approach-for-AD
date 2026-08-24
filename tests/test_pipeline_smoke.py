"""End-to-end smoke test: the full data path on synthetic data, no OASIS needed.

Exercises manifest loading -> volume processing -> subject-wise splitting ->
augmentation (post-split only), and checks the specific properties called
out in the migration scope: shape/range, deterministic seeds, malformed
contracts, temporal edge cases, no subject overlap, and augmentation only
after splitting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal_ad.data.augmentation import augment_volume
from multimodal_ad.data.labeling import classify_diagnosis_row, smooth_temporal_labels
from multimodal_ad.data.manifest import ManifestValidationError, ScanManifest
from multimodal_ad.data.splits import assert_no_subject_leakage, subject_train_test_split
from multimodal_ad.data.synthetic import SyntheticDatasetConfig, generate_synthetic_dataset
from multimodal_ad.data.volumes import ProcessingConfig, process_scan


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> ScanManifest:
    config = SyntheticDatasetConfig(n_subjects=12, seed=1234)
    return generate_synthetic_dataset(tmp_path / "synthetic", config)


def test_full_pipeline_shape_and_range(synthetic_manifest: ScanManifest) -> None:
    processing_config = ProcessingConfig(n_frames=10, image_size=32)
    for record in synthetic_manifest.records[:3]:
        volume = process_scan(record.file_path, record.modality, processing_config)
        assert volume.shape == (32, 32, 10)
        assert volume.dtype == np.float32
        assert 0.0 <= volume.min() <= volume.max() <= 1.0


def test_full_pipeline_is_deterministic_end_to_end(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(n_subjects=6, seed=99)
    manifest_a = generate_synthetic_dataset(tmp_path / "a", config)
    manifest_b = generate_synthetic_dataset(tmp_path / "b", config)

    processing_config = ProcessingConfig(n_frames=8, image_size=16)
    volume_a = process_scan(
        manifest_a.records[0].file_path, manifest_a.records[0].modality, processing_config
    )
    volume_b = process_scan(
        manifest_b.records[0].file_path, manifest_b.records[0].modality, processing_config
    )
    np.testing.assert_array_equal(volume_a, volume_b)


def test_full_pipeline_split_has_no_subject_overlap(synthetic_manifest: ScanManifest) -> None:
    df = synthetic_manifest.to_dataframe()
    train_df, test_df = subject_train_test_split(df, test_size=0.3, seed=1234)
    assert_no_subject_leakage(train_df, test_df)  # must not raise
    assert set(train_df["subject_id"]) & set(test_df["subject_id"]) == set()


def test_augmentation_applied_only_after_split_does_not_leak_subjects(
    synthetic_manifest: ScanManifest,
) -> None:
    """Augmenting only the train split must not introduce test-subject data."""
    df = synthetic_manifest.to_dataframe()
    train_df, test_df = subject_train_test_split(df, test_size=0.3, seed=1234)

    rng = np.random.default_rng(1234)
    processing_config = ProcessingConfig(n_frames=8, image_size=16)
    train_records = ScanManifest.from_dataframe(train_df).records

    augmented_subjects = set()
    for record in train_records[:3]:
        volume = process_scan(record.file_path, record.modality, processing_config)
        augment_volume(volume, rng)
        augmented_subjects.add(record.subject_id)

    # Every augmented sample's subject must have come from the train split,
    # never the test split.
    assert augmented_subjects <= set(train_df["subject_id"])
    assert augmented_subjects.isdisjoint(set(test_df["subject_id"]))


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    bad_csv = tmp_path / "manifest.csv"
    pd.DataFrame({"subject_id": ["A"], "session_id": ["A_MR_d1"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ManifestValidationError):
        ScanManifest.from_csv(bad_csv)


def test_temporal_smoothing_edge_case_single_visit_subject() -> None:
    """A subject with exactly one visit can never be "surrounded"; must stay as-is."""
    labels = pd.Series([False])
    subjects = pd.Series(["ONLY"])
    days = pd.Series([0])
    smoothed = smooth_temporal_labels(labels, subjects, days)
    assert smoothed.tolist() == [False]


def test_temporal_smoothing_edge_case_all_false_never_corrected() -> None:
    labels = pd.Series([False, False, False])
    subjects = pd.Series(["A", "A", "A"])
    days = pd.Series([0, 10, 20])
    smoothed = smooth_temporal_labels(labels, subjects, days)
    assert smoothed.tolist() == [False, False, False]


def test_diagnosis_classification_uncertain_treated_as_demented() -> None:
    # Retained paper-relevant decision: "we classify uncertain cases as sick".
    assert classify_diagnosis_row([None, "uncertain dementia, possible"]) is True
