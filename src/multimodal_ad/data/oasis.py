"""Adapter for a locally provided OASIS-3 directory layout.

**This module never downloads, mirrors, or redistributes OASIS-3 data.** It
only reads CSV metadata and scan files that the user has already obtained
under OASIS-3's own data use agreement and placed on local disk. See
https://sites.wustl.edu/oasisbrains/ for access terms.

OASIS-3 session identifiers look like `OAS30001_MR_d0129`: subject id,
modality tag, and a `d<day-offset>` field giving days since the subject's
baseline visit (not a calendar date). `parse_session_id` is the single
place this format is decoded, shared by this adapter and
`data.synthetic` (whose generator produces ids in the same format so it
exercises the same parsing path).

**Open ambiguity**: the legacy notebooks derive `Date`/`Subject`/`Tracer`
via positional string-splitting of several different identifier columns
across five separate OASIS CSVs (`subjects`, `clinical-data`, `pet`, `pup`,
`mri`), without validating the split. This adapter intentionally only
commits to the one identifier format that is unambiguous from the
notebooks (`parse_session_id`, above) plus a minimal, explicit CSV/column
contract (`OasisLayout`). The exact column names/schemas of each real
OASIS-3 CSV export were not recoverable from the notebooks alone and must
be confirmed against a real local export before this adapter is treated as
schema-complete (see `docs/legacy-notebooks-inventory.md`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_SESSION_ID_PATTERN = re.compile(
    r"^(?P<subject_id>[A-Za-z0-9]+)_(?P<modality_tag>[A-Za-z0-9]+)_d(?P<day_offset>\d+)$"
)


class OasisLayoutError(FileNotFoundError):
    """Raised when a locally provided OASIS-3 export is missing expected files."""


class SessionIdError(ValueError):
    """Raised when a session identifier does not match the expected OASIS format."""


@dataclass(frozen=True, slots=True)
class ParsedSessionId:
    subject_id: str
    modality_tag: str
    day_offset: int


def parse_session_id(session_id: str) -> ParsedSessionId:
    """Parse an OASIS-3 session id like `OAS30001_MR_d0129`.

    Raises `SessionIdError` if `session_id` does not match the expected
    `<subject>_<modality>_d<day_offset>` shape.
    """
    match = _SESSION_ID_PATTERN.match(session_id)
    if match is None:
        raise SessionIdError(
            f"session id does not match <subject>_<modality>_d<days>: {session_id!r}"
        )
    try:
        day_offset = int(match.group("day_offset"))
    except ValueError as exc:
        raise SessionIdError(f"invalid day offset in session id: {session_id!r}") from exc
    return ParsedSessionId(
        subject_id=match.group("subject_id"),
        modality_tag=match.group("modality_tag"),
        day_offset=day_offset,
    )


@dataclass(frozen=True, slots=True)
class OasisLayout:
    """Expected file layout of a locally provided OASIS-3 export.

    File names are configurable defaults, not a hardcoded assumption about
    the user's export naming; override any of them to match a real export.
    """

    root: Path
    subjects_csv: str = "subjects.csv"
    clinical_csv: str = "clinical-data.csv"
    pet_csv: str = "pet.csv"
    pup_csv: str = "pup.csv"
    mri_csv: str = "mri.csv"
    scans_dir: str = "scans"

    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def required_csvs(self) -> tuple[str, ...]:
        return (self.subjects_csv, self.clinical_csv, self.pet_csv, self.pup_csv, self.mri_csv)


def validate_layout(layout: OasisLayout) -> None:
    """Check that all expected local files/directories exist.

    Raises `OasisLayoutError` (never downloads anything) if the layout is
    incomplete.
    """
    if not layout.root.is_dir():
        raise OasisLayoutError(
            f"OASIS root does not exist or is not a directory: {layout.root}. "
            "This adapter reads a locally provided OASIS-3 export; it does not "
            "download data. Obtain access at https://sites.wustl.edu/oasisbrains/."
        )
    missing = [name for name in layout.required_csvs if not layout.path(name).exists()]
    if missing:
        raise OasisLayoutError(f"OASIS root {layout.root} is missing expected files: {missing}.")
    if not layout.path(layout.scans_dir).is_dir():
        raise OasisLayoutError(
            f"expected scans directory not found: {layout.path(layout.scans_dir)}"
        )


def load_clinical_table(layout: OasisLayout) -> pl.DataFrame:
    """Read the locally provided clinical-visits CSV.

    Expected (minimum) columns: `subject_id`, `day_offset`, and a diagnosis
    column suitable for `data.labeling.classify_diagnosis_row`.
    """
    validate_layout(layout)
    return pl.read_csv(layout.path(layout.clinical_csv))


def load_scan_index(layout: OasisLayout, csv_name: str) -> pl.DataFrame:
    """Read a locally provided per-modality scan index CSV (`mri.csv`/`pet.csv`).

    Expected (minimum) column: `session_id`, parseable by `parse_session_id`.
    """
    validate_layout(layout)
    return pl.read_csv(layout.path(csv_name))
