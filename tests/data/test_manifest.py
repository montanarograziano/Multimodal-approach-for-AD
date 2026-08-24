"""Tests for `multimodal_ad.data.manifest`."""

from pathlib import Path

import pandas as pd
import pytest

from multimodal_ad.data.manifest import (
    REQUIRED_COLUMNS,
    ManifestValidationError,
    Modality,
    ScanManifest,
    ScanRecord,
)


def _valid_row(tmp_file: Path, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "subject_id": "OAS30001",
        "session_id": "OAS30001_MR_d0001",
        "modality": "MRI",
        "day_offset": 1,
        "label": 1,
        "file_path": str(tmp_file),
        "tracer": None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def scan_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.nii.gz"
    path.write_bytes(b"not a real nifti, existence is all that's checked")
    return path


def test_round_trip_dataframe(scan_file: Path) -> None:
    df = pd.DataFrame([_valid_row(scan_file)])
    manifest = ScanManifest.from_dataframe(df)
    assert len(manifest) == 1
    record = manifest.records[0]
    assert record.subject_id == "OAS30001"
    assert record.modality is Modality.MRI
    assert record.label == 1

    round_tripped = manifest.to_dataframe()
    assert list(round_tripped.columns) == list(REQUIRED_COLUMNS) + ["tracer"]


def test_missing_required_column_raises(scan_file: Path) -> None:
    df = pd.DataFrame([_valid_row(scan_file)]).drop(columns=["label"])
    with pytest.raises(ManifestValidationError, match="missing required columns"):
        ScanManifest.from_dataframe(df)


def test_duplicate_session_id_raises(scan_file: Path) -> None:
    df = pd.DataFrame([_valid_row(scan_file), _valid_row(scan_file)])
    with pytest.raises(ManifestValidationError, match="duplicate session_id"):
        ScanManifest.from_dataframe(df)


def test_unknown_modality_raises(scan_file: Path) -> None:
    df = pd.DataFrame([_valid_row(scan_file, modality="XRAY")])
    with pytest.raises(ManifestValidationError, match="invalid modality"):
        ScanManifest.from_dataframe(df)


def test_missing_file_raises_by_default(tmp_path: Path) -> None:
    df = pd.DataFrame([_valid_row(tmp_path / "does-not-exist.nii.gz")])
    with pytest.raises(ManifestValidationError, match="does not exist"):
        ScanManifest.from_dataframe(df)


def test_missing_file_allowed_when_not_required(tmp_path: Path) -> None:
    df = pd.DataFrame([_valid_row(tmp_path / "does-not-exist.nii.gz")])
    manifest = ScanManifest.from_dataframe(df, require_files_exist=False)
    assert len(manifest) == 1


def test_invalid_label_raises(scan_file: Path) -> None:
    df = pd.DataFrame([_valid_row(scan_file, label="not-a-label")])
    with pytest.raises(ManifestValidationError, match="label"):
        ScanManifest.from_dataframe(df)


def test_scan_record_rejects_out_of_range_label(scan_file: Path) -> None:
    with pytest.raises(ManifestValidationError, match="label must be"):
        ScanRecord(
            subject_id="OAS30001",
            session_id="OAS30001_MR_d0001",
            modality=Modality.MRI,
            day_offset=1,
            label=2,
            file_path=scan_file,
        )


def test_filter_modality(scan_file: Path) -> None:
    df = pd.DataFrame(
        [
            _valid_row(scan_file, session_id="s-mri", modality="MRI"),
            _valid_row(scan_file, session_id="s-pet", modality="PET"),
        ]
    )
    manifest = ScanManifest.from_dataframe(df)
    mri_only = manifest.filter_modality(Modality.MRI)
    assert len(mri_only) == 1
    assert mri_only.records[0].modality is Modality.MRI


def test_csv_round_trip(scan_file: Path, tmp_path: Path) -> None:
    df = pd.DataFrame([_valid_row(scan_file)])
    manifest = ScanManifest.from_dataframe(df)
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path)
    reloaded = ScanManifest.from_csv(csv_path)
    assert reloaded.records == manifest.records
