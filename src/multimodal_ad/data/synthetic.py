"""Deterministic synthetic NIfTI + manifest generator.

Produces a small, fake-but-structurally-realistic dataset (NIfTI volumes
with a synthetic "brain" blob, plus a matching manifest CSV) so the full
data path (`oasis`/`manifest` -> `labeling` -> `volumes` -> `augmentation`
-> `splits`) is exercisable end to end without real OASIS-3 data, which
this project cannot download or redistribute (see `data.oasis`).

Session ids are generated in the same `<subject>_<modality>_d<days>` shape
`data.oasis.parse_session_id` expects, so the synthetic data exercises the
same parsing path real OASIS data would.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from multimodal_ad.data.manifest import Modality, ScanManifest, ScanRecord

#: Retained from the legacy notebooks (`RANDOM_SEED = 1234`).
DEFAULT_SEED = 1234


@dataclass(frozen=True, slots=True)
class SyntheticDatasetConfig:
    n_subjects: int = 20
    sessions_per_subject_range: tuple[int, int] = (1, 4)  # inclusive
    volume_shape: tuple[int, int, int] = (64, 64, 64)
    background_mean: float = 50.0
    background_std: float = 5.0
    brain_mean: float = 500.0
    brain_std: float = 50.0
    day_gap_range: tuple[int, int] = (30, 400)  # inclusive, days between sessions
    modality: Modality = Modality.MRI
    seed: int = DEFAULT_SEED


def _synthetic_volume(rng: np.random.Generator, config: SyntheticDatasetConfig) -> np.ndarray:
    """A noisy background with a brighter central cube standing in for a brain.

    The central cube gives `volumes.find_brain_bounding_box`'s Otsu
    thresholding a real foreground to detect, so the synthetic dataset
    exercises the same cropping logic real scans would.
    """
    volume = rng.normal(config.background_mean, config.background_std, config.volume_shape)
    shape = np.array(config.volume_shape)
    margin = shape // 4
    lo, hi = margin, shape - margin
    brain_shape = tuple(hi - lo)
    volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = rng.normal(
        config.brain_mean, config.brain_std, brain_shape
    )
    return np.clip(volume, 0, None).astype(np.float32)


def generate_synthetic_dataset(
    output_dir: Path, config: SyntheticDatasetConfig | None = None
) -> ScanManifest:
    """Write deterministic synthetic NIfTI volumes and return their manifest.

    Volumes are written under `output_dir` and a `manifest.csv` alongside
    them; both are also returned as an in-memory `ScanManifest`. Calling
    this twice with the same `config` (including `seed`) produces
    byte-identical output.
    """
    config = config or SyntheticDatasetConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    records: list[ScanRecord] = []
    for subject_idx in range(config.n_subjects):
        subject_id = f"SYNTH{subject_idx:04d}"
        # A subject's disease label is fixed across their sessions here; the
        # per-visit noise that `labeling.smooth_temporal_labels` corrects is
        # exercised by tests directly on synthetic label sequences instead,
        # keeping this generator's contract simple.
        label = int(rng.integers(0, 2))
        n_sessions = int(
            rng.integers(
                config.sessions_per_subject_range[0], config.sessions_per_subject_range[1] + 1
            )
        )

        day_offset = 0
        for _session_idx in range(n_sessions):
            day_offset += int(rng.integers(config.day_gap_range[0], config.day_gap_range[1] + 1))
            session_id = f"{subject_id}_{config.modality.value}_d{day_offset:04d}"
            file_path = output_dir / f"{session_id}.nii.gz"

            volume = _synthetic_volume(rng, config)
            nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), file_path)

            records.append(
                ScanRecord(
                    subject_id=subject_id,
                    session_id=session_id,
                    modality=config.modality,
                    day_offset=day_offset,
                    label=label,
                    file_path=file_path,
                )
            )

    manifest = ScanManifest(records)
    manifest.to_csv(output_dir / "manifest.csv")
    return manifest


def load_synthetic_manifest(output_dir: Path) -> pd.DataFrame:
    """Read back a manifest CSV previously written by `generate_synthetic_dataset`."""
    return pd.read_csv(output_dir / "manifest.csv")
