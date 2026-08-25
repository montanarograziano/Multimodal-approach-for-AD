"""Tests for `multimodal_ad.data.synthetic`."""

from pathlib import Path

import numpy as np

from multimodal_ad.data.manifest import ScanManifest
from multimodal_ad.data.synthetic import (
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
    load_synthetic_manifest,
)
from multimodal_ad.data.volumes import load_volume


def test_generate_synthetic_dataset_writes_files_and_manifest(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(n_subjects=5, seed=1234)
    manifest = generate_synthetic_dataset(tmp_path, config)

    assert len(manifest) > 0
    assert (tmp_path / "manifest.csv").exists()
    for record in manifest.records:
        assert record.file_path.exists()

    subjects = {r.subject_id for r in manifest.records}
    assert len(subjects) == 5


def test_generate_synthetic_dataset_manifest_round_trips(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(n_subjects=3, seed=1234)
    generate_synthetic_dataset(tmp_path, config)

    df = load_synthetic_manifest(tmp_path)
    reloaded = ScanManifest.from_dataframe(df)
    assert len(reloaded) > 0


def test_generate_synthetic_dataset_is_deterministic(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(n_subjects=4, seed=1234)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    manifest_a = generate_synthetic_dataset(out_a, config)
    manifest_b = generate_synthetic_dataset(out_b, config)

    df_a = manifest_a.to_dataframe().drop(columns=["file_path"])
    df_b = manifest_b.to_dataframe().drop(columns=["file_path"])
    assert df_a.equals(df_b)

    volume_a = load_volume(manifest_a.records[0].file_path)
    volume_b = load_volume(manifest_b.records[0].file_path)
    np.testing.assert_array_equal(volume_a, volume_b)


def test_generate_synthetic_dataset_different_seeds_differ(tmp_path: Path) -> None:
    manifest_a = generate_synthetic_dataset(
        tmp_path / "a", SyntheticDatasetConfig(n_subjects=4, seed=1)
    )
    manifest_b = generate_synthetic_dataset(
        tmp_path / "b", SyntheticDatasetConfig(n_subjects=4, seed=2)
    )
    labels_a = [r.label for r in manifest_a.records]
    labels_b = [r.label for r in manifest_b.records]
    assert labels_a != labels_b


def test_generate_synthetic_dataset_no_duplicate_session_ids(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(n_subjects=10, seed=7)
    manifest = generate_synthetic_dataset(tmp_path, config)
    session_ids = [r.session_id for r in manifest.records]
    assert len(session_ids) == len(set(session_ids))


def test_generate_synthetic_dataset_sessions_increase_in_day_offset(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(n_subjects=6, seed=3)
    manifest = generate_synthetic_dataset(tmp_path, config)

    by_subject: dict[str, list[int]] = {}
    for record in manifest.records:
        by_subject.setdefault(record.subject_id, []).append(record.day_offset)

    for offsets in by_subject.values():
        assert offsets == sorted(offsets)
        assert len(offsets) == len(set(offsets))
