"""Tests for `multimodal_ad.data.oasis`."""

from pathlib import Path

import polars as pl
import pytest

from multimodal_ad.data.oasis import (
    OasisLayout,
    OasisLayoutError,
    SessionIdError,
    load_clinical_table,
    load_scan_index,
    parse_session_id,
    validate_layout,
)


def test_parse_session_id_extracts_fields() -> None:
    parsed = parse_session_id("OAS30001_MR_d0129")
    assert parsed.subject_id == "OAS30001"
    assert parsed.modality_tag == "MR"
    assert parsed.day_offset == 129


def test_parse_session_id_rejects_malformed_ids() -> None:
    with pytest.raises(SessionIdError):
        parse_session_id("not-a-valid-session-id")
    with pytest.raises(SessionIdError):
        parse_session_id("OAS30001_MR_129")  # missing 'd' prefix


def _make_layout(root: Path, *, complete: bool = True) -> OasisLayout:
    layout = OasisLayout(root=root)
    if complete:
        for name in layout.required_csvs:
            (root / name).write_text("subject_id,day_offset\n")
        (root / layout.scans_dir).mkdir()
    return layout


def test_validate_layout_passes_when_complete(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    validate_layout(layout)  # must not raise


def test_validate_layout_raises_on_missing_root(tmp_path: Path) -> None:
    layout = OasisLayout(root=tmp_path / "does-not-exist")
    with pytest.raises(OasisLayoutError, match="does not exist"):
        validate_layout(layout)


def test_validate_layout_raises_on_missing_csv(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    (tmp_path / layout.clinical_csv).unlink()
    with pytest.raises(OasisLayoutError, match="missing expected files"):
        validate_layout(layout)


def test_validate_layout_raises_on_missing_scans_dir(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    (tmp_path / layout.scans_dir).rmdir()
    with pytest.raises(OasisLayoutError, match="scans directory"):
        validate_layout(layout)


def test_load_clinical_table_reads_local_csv(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    (tmp_path / layout.clinical_csv).write_text("subject_id,day_offset\nOAS1,0\n")
    df = load_clinical_table(layout)
    assert df.get_column("subject_id").to_list() == ["OAS1"]


def test_load_scan_index_reads_local_csv(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    (tmp_path / layout.mri_csv).write_text("session_id\nOAS1_MR_d0001\n")
    df = load_scan_index(layout, layout.mri_csv)
    assert isinstance(df, pl.DataFrame)
    assert df.get_column("session_id")[0] == "OAS1_MR_d0001"


def test_load_clinical_table_never_downloads_missing_data(tmp_path: Path) -> None:
    layout = OasisLayout(root=tmp_path / "missing")
    with pytest.raises(OasisLayoutError):
        load_clinical_table(layout)
    # No network/download side effect is possible: the failure is a local
    # filesystem check only. Assert the (still nonexistent) root was not
    # created as a side effect of "trying" to fetch anything.
    assert not (tmp_path / "missing").exists()
