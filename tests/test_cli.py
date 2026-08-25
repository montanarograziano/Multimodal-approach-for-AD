"""Tests for the `multimodal_ad.cli` synthetic quickstart."""

from pathlib import Path

from multimodal_ad.cli import build_parser, main


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.n_subjects == 20
    assert args.seed == 1234


def test_main_runs_end_to_end(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "synthetic"
    main(
        [
            "--output-dir",
            str(output_dir),
            "--n-subjects",
            "6",
            "--n-frames",
            "8",
            "--image-size",
            "32",
            "--seed",
            "1234",
        ]
    )
    captured = capsys.readouterr()
    assert "generated" in captured.out
    assert "split:" in captured.out
    assert "processed" in captured.out
    assert (output_dir / "manifest.csv").exists()
