"""Smoke tests for the multimodal_ad package foundation."""

import multimodal_ad


def test_version_is_set() -> None:
    assert multimodal_ad.__version__ == "0.1.0"


def test_package_importable() -> None:
    assert multimodal_ad.__doc__ is not None
