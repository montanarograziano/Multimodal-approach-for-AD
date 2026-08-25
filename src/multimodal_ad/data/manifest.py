"""Typed scan manifest: the canonical data contract for MRI/PET metadata.

A `ScanManifest` is the single structure the rest of the data pipeline
consumes and produces, decoupling "how did we get this metadata" (a local
OASIS-3 export, the synthetic generator, ...) from "what a scan record looks
like" (loading, labeling, splitting, augmentation).

OASIS-3 session identifiers encode a *day offset* since a subject's baseline
visit (e.g. `d0236`), not a calendar date. We keep that as an integer
`day_offset` rather than inventing a calendar date: it is the field the
legacy notebooks actually use for nearest-visit matching and temporal
ordering (see `docs/legacy-notebooks-inventory.md`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd


class Modality(StrEnum):
    """Imaging modality. `StrEnum` keeps CSV round-tripping trivial."""

    MRI = "MRI"
    PET = "PET"


class ManifestValidationError(ValueError):
    """Raised when scan manifest data violates the typed data contract."""


REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "session_id",
    "modality",
    "day_offset",
    "label",
    "file_path",
)


@dataclass(frozen=True, slots=True)
class ScanRecord:
    """One validated MRI or PET scan and its metadata."""

    subject_id: str
    session_id: str
    modality: Modality
    day_offset: int
    label: int | None  # 1 = demented, 0 = not demented, None = unlabeled
    file_path: Path
    tracer: str | None = None

    def __post_init__(self) -> None:
        if self.label is not None and self.label not in (0, 1):
            raise ManifestValidationError(
                f"{self.session_id}: label must be 0, 1, or None, got {self.label!r}"
            )
        if not self.subject_id:
            raise ManifestValidationError(f"{self.session_id}: subject_id must be non-empty")


@dataclass(slots=True)
class ScanManifest:
    """An ordered collection of `ScanRecord`s with (de)serialization helpers."""

    records: list[ScanRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ScanRecord]:
        return iter(self.records)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, require_files_exist: bool = True) -> ScanManifest:
        """Build and validate a manifest from a DataFrame matching the data contract.

        Raises `ManifestValidationError` on missing columns, duplicate
        session ids, unknown modalities, invalid labels, or (when
        `require_files_exist`) missing scan files.
        """
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_columns:
            raise ManifestValidationError(
                f"manifest is missing required columns: {missing_columns}"
            )

        records: list[ScanRecord] = []
        seen_sessions: set[str] = set()
        has_tracer = "tracer" in df.columns
        for row_dict in df.to_dict(orient="records"):
            session_id = str(row_dict["session_id"])
            if session_id in seen_sessions:
                raise ManifestValidationError(f"duplicate session_id in manifest: {session_id!r}")
            seen_sessions.add(session_id)

            modality_raw = str(row_dict["modality"]).upper()
            try:
                modality = Modality(modality_raw)
            except ValueError as exc:
                valid = [m.value for m in Modality]
                raise ManifestValidationError(
                    f"{session_id}: invalid modality {row_dict['modality']!r}, "
                    f"expected one of {valid}"
                ) from exc

            file_path = Path(str(row_dict["file_path"]))
            if require_files_exist and not file_path.exists():
                raise ManifestValidationError(
                    f"{session_id}: file_path does not exist: {file_path}"
                )

            label_raw = row_dict["label"]
            try:
                label = None if pd.isna(label_raw) else int(label_raw)
            except (TypeError, ValueError) as exc:
                raise ManifestValidationError(
                    f"{session_id}: label must be numeric or missing, got {label_raw!r}"
                ) from exc

            day_offset_raw = row_dict["day_offset"]
            if pd.isna(day_offset_raw):
                raise ManifestValidationError(f"{session_id}: day_offset must not be missing")
            try:
                day_offset = int(day_offset_raw)
            except (TypeError, ValueError) as exc:
                raise ManifestValidationError(
                    f"{session_id}: day_offset must be an integer, got {day_offset_raw!r}"
                ) from exc

            tracer_raw = row_dict.get("tracer") if has_tracer else None
            tracer = None if tracer_raw is None or pd.isna(tracer_raw) else str(tracer_raw)

            records.append(
                ScanRecord(
                    subject_id=str(row_dict["subject_id"]),
                    session_id=session_id,
                    modality=modality,
                    day_offset=day_offset,
                    label=label,
                    file_path=file_path,
                    tracer=tracer,
                )
            )
        return cls(records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "subject_id": [r.subject_id for r in self.records],
                "session_id": [r.session_id for r in self.records],
                "modality": [r.modality.value for r in self.records],
                "day_offset": [r.day_offset for r in self.records],
                "label": [r.label for r in self.records],
                "file_path": [str(r.file_path) for r in self.records],
                "tracer": [r.tracer for r in self.records],
            }
        )

    @classmethod
    def from_csv(cls, path: Path, *, require_files_exist: bool = True) -> ScanManifest:
        return cls.from_dataframe(pd.read_csv(path), require_files_exist=require_files_exist)

    def to_csv(self, path: Path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def filter_modality(self, modality: Modality) -> ScanManifest:
        return ScanManifest([r for r in self.records if r.modality is modality])

    def filter_labeled(self) -> ScanManifest:
        return ScanManifest([r for r in self.records if r.label is not None])
