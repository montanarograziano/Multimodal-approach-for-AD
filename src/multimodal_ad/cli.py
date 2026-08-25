"""Synthetic data pipeline quickstart CLI.

Generates a deterministic synthetic dataset and runs it through the full
data path (manifest -> volume processing -> subject-wise split), so the
pipeline can be exercised end to end without OASIS-3 data. Intentionally a
thin `argparse` wrapper, not a framework: `python -m multimodal_ad.cli
--help` for options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_ad.data.manifest import ScanManifest
from multimodal_ad.data.splits import subject_train_test_split
from multimodal_ad.data.synthetic import (
    DEFAULT_SEED,
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
)
from multimodal_ad.data.volumes import ProcessingConfig, process_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-ad-quickstart",
        description=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic-data"),
        help="where to write synthetic scans",
    )
    parser.add_argument("--n-subjects", type=int, default=20, help="number of synthetic subjects")
    parser.add_argument("--n-frames", type=int, default=16, help="central axial frames to process")
    parser.add_argument("--image-size", type=int, default=64, help="output frame size (pixels)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic RNG seed")
    return parser


def run(args: argparse.Namespace) -> None:
    dataset_config = SyntheticDatasetConfig(n_subjects=args.n_subjects, seed=args.seed)
    manifest = generate_synthetic_dataset(args.output_dir, dataset_config)
    print(f"generated {len(manifest)} synthetic scans across {args.n_subjects} subjects")

    df = manifest.to_dataframe()
    train_df, test_df = subject_train_test_split(df, seed=args.seed)
    print(f"split: {len(train_df)} train scans, {len(test_df)} test scans (no subject overlap)")

    processing_config = ProcessingConfig(n_frames=args.n_frames, image_size=args.image_size)
    reloaded = ScanManifest.from_dataframe(train_df)
    first_record = reloaded.records[0]
    volume = process_scan(first_record.file_path, first_record.modality, processing_config)
    print(
        f"processed {first_record.session_id}: shape={volume.shape}, "
        f"range=[{volume.min():.3f}, {volume.max():.3f}]"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
