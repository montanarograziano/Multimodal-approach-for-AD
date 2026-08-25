"""Tests for `multimodal_ad.models.architecture`."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from multimodal_ad.models.architecture import (  # noqa: E402
    Cnn3DConfig,
    build_3d_cnn,
    build_fusion_model,
    build_mri_input_model,
    build_pet_input_model,
    extract_feature_extractor,
)

#: Smallest cubic size that survives four (Conv3D(k=3, valid) -> MaxPool3D(2))
#: blocks without a spatial dimension collapsing to <1 or <2 (see
#: `docs/legacy-notebooks-inventory.md` for the retained topology); kept
#: tiny for CPU-fast tests, real training uses (128, 128, 50).
_TINY = 48


def _tiny_config(**overrides: object) -> Cnn3DConfig:
    base: dict[str, object] = {"width": _TINY, "height": _TINY, "depth": _TINY}
    base.update(overrides)
    return Cnn3DConfig(**base)  # type: ignore[arg-type]


def test_build_3d_cnn_default_topology_matches_notebook_get_3d_model() -> None:
    model = build_3d_cnn(_tiny_config(name="default"))
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv3D)]
    assert [layer.filters for layer in conv_layers] == [64, 64, 128, 256]
    assert all(layer.kernel_size == (3, 3, 3) for layer in conv_layers)
    assert all(layer.trainable for layer in conv_layers)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    assert dense_layers[0].units == 512
    assert dense_layers[0].trainable
    assert dense_layers[-1].units == 1
    assert dense_layers[-1].activation.__name__ == "sigmoid"


def test_build_3d_cnn_forward_pass_shape_and_range() -> None:
    model = build_3d_cnn(_tiny_config(name="fwd"))
    x = np.random.default_rng(0).random((2, _TINY, _TINY, _TINY, 1)).astype("float32")
    # Keras 3's `predict` infers `verbose`'s type as `str` from its `"auto"`
    # default; it accepts (and documents) an int verbosity level at runtime.
    predictions = model.predict(x, verbose=0)  # pyright: ignore[reportArgumentType]
    assert predictions.shape == (2, 1)
    assert np.all(np.isfinite(predictions))
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))


def test_build_3d_cnn_freeze_conv_flag_freezes_conv_and_batchnorm_only() -> None:
    model = build_3d_cnn(_tiny_config(freeze_conv=True, freeze_dense=False, name="frozen"))
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv3D | tf.keras.layers.BatchNormalization):
            assert not layer.trainable
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    assert dense_layers[0].trainable  # get_3d_frozen leaves the Dense(512) head trainable


def test_build_mri_input_model_freezes_conv_and_dense_and_uses_named_layers() -> None:
    model = build_mri_input_model(_tiny_config(), name="mri")
    assert [i.name for i in model.inputs] == ["input_x"]
    assert "drop_x" in [layer.name for layer in model.layers]
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    assert not dense_layers[0].trainable  # get_3d_x also freezes the Dense(512) head


def test_build_pet_input_model_names_input_dropout_and_output() -> None:
    model = build_pet_input_model(_tiny_config(), name="pet")
    assert [i.name for i in model.inputs] == ["input_y"]
    layer_names = [layer.name for layer in model.layers]
    assert "drop_y" in layer_names
    assert "out_y" in layer_names


def test_extract_feature_extractor_exposes_penultimate_dense_layer() -> None:
    model = build_3d_cnn(_tiny_config(name="feat"))
    features = extract_feature_extractor(model)
    assert features.output_shape == (None, 512)


def test_build_fusion_model_accepts_two_inputs_and_shape() -> None:
    config = _tiny_config()
    mri_model = build_mri_input_model(config, name="mri")
    pet_model = build_pet_input_model(config, name="pet")
    fusion = build_fusion_model(mri_model, pet_model, name="merged")

    assert len(fusion.inputs) == 2
    x_mri = np.random.default_rng(1).random((2, _TINY, _TINY, _TINY, 1)).astype("float32")
    x_pet = np.random.default_rng(2).random((2, _TINY, _TINY, _TINY, 1)).astype("float32")
    predictions = fusion.predict([x_mri, x_pet], verbose=0)  # pyright: ignore[reportArgumentType]
    assert predictions.shape == (2, 1)
    assert np.all(np.isfinite(predictions))


def test_build_fusion_model_head_is_dense_128_not_dense_4() -> None:
    """Canonical fusion head per module docstring: the first, non-buggy `get_merged()`."""
    config = _tiny_config()
    mri_model = build_mri_input_model(config, name="mri2")
    pet_model = build_pet_input_model(config, name="pet2")
    fusion = build_fusion_model(mri_model, pet_model, name="merged2")
    dense_layers = [layer for layer in fusion.layers if isinstance(layer, tf.keras.layers.Dense)]
    # dense_layers order: [mri Dense(512), pet Dense(512), fusion head Dense(128), output Dense(1)]
    assert dense_layers[-2].units == 128
