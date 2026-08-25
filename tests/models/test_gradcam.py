"""Tests for `multimodal_ad.models.gradcam`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from multimodal_ad.models.architecture import Cnn3DConfig, build_3d_cnn  # noqa: E402
from multimodal_ad.models.gradcam import (  # noqa: E402
    last_conv_layer_name,
    make_gradcam_heatmap,
    unwrap_output_activation,
)

_TINY = 48


if TYPE_CHECKING:
    import keras as _keras_types


def _model() -> _keras_types.Model:
    return build_3d_cnn(Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="gradcam"))


def test_last_conv_layer_name_finds_final_conv3d() -> None:
    model = _model()
    name = last_conv_layer_name(model)
    layer = model.get_layer(name)
    assert layer.filters == 256  # the notebook's final Conv3D block


def test_last_conv_layer_name_raises_for_missing_layer_type() -> None:
    model = _model()
    with pytest.raises(ValueError, match="Conv2D"):
        last_conv_layer_name(model, layer_type=keras.layers.Conv2D)


def test_make_gradcam_heatmap_is_finite_and_in_unit_range() -> None:
    model = _model()
    unwrap_output_activation(model)
    layer_name = last_conv_layer_name(model)

    volume = np.random.default_rng(0).random((1, _TINY, _TINY, _TINY, 1)).astype("float32")
    heatmap = make_gradcam_heatmap(volume, model, layer_name)

    assert heatmap.ndim == 3  # (D, H, W) spatial map from the last Conv3D block
    assert np.all(np.isfinite(heatmap))
    assert np.all((heatmap >= 0.0) & (heatmap <= 1.0))


def test_unwrap_output_activation_makes_output_layer_linear() -> None:
    model = _model()
    unwrap_output_activation(model)
    assert model.layers[-1].activation is keras.activations.linear
