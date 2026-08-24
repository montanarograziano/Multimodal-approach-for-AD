"""Tests for `multimodal_ad.models.training`."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from multimodal_ad.models.architecture import Cnn3DConfig, build_3d_cnn  # noqa: E402
from multimodal_ad.models.training import (  # noqa: E402
    TrainingConfig,
    concatenate_modalities,
    load_model,
    save_model,
    seed_everything,
    train_model,
)

_TINY = 48


def _tiny_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, _TINY, _TINY, _TINY, 1)).astype("float32")
    y = (np.arange(n) % 2).astype("float32")
    return x, y


def test_seed_everything_makes_model_init_deterministic() -> None:
    seed_everything(1234)
    model_a = build_3d_cnn(Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="a"))
    seed_everything(1234)
    model_b = build_3d_cnn(Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="b"))

    weights_a = model_a.get_layer(index=1).get_weights()[0]
    weights_b = model_b.get_layer(index=1).get_weights()[0]
    np.testing.assert_array_equal(weights_a, weights_b)


def test_concatenate_modalities_pools_arrays() -> None:
    x_a, y_a = _tiny_data(3, seed=1)
    x_b, y_b = _tiny_data(2, seed=2)
    x, y = concatenate_modalities(x_a, y_a, x_b, y_b)
    assert x.shape == (5, _TINY, _TINY, _TINY, 1)
    assert y.shape == (5,)


def test_train_model_runs_one_tiny_step_and_evaluates(tmp_path: Path) -> None:
    x_train, y_train = _tiny_data(4, seed=10)
    x_val, y_val = _tiny_data(4, seed=11)
    model = build_3d_cnn(Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="train"))

    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        checkpoint_path=tmp_path / "checkpoint.keras",
        early_stopping_baseline=None,  # allow the 1-epoch run to always "count"
        verbose=0,
    )
    result = train_model(model, x_train, y_train, x_val, y_val, config)

    assert len(result.history.history["loss"]) == 1
    assert np.isfinite(result.val_loss)
    assert 0.0 <= result.val_accuracy <= 1.0


def test_save_and_load_keras_model_preserves_predictions(tmp_path: Path) -> None:
    seed_everything(1234)
    model = build_3d_cnn(Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="save"))
    x, _ = _tiny_data(2, seed=42)
    predictions_before = model.predict(x, verbose=0)

    path = tmp_path / "model.keras"
    save_model(model, path)
    reloaded = load_model(path)
    predictions_after = reloaded.predict(x, verbose=0)

    np.testing.assert_allclose(predictions_before, predictions_after)


def test_save_and_load_legacy_h5_preserves_predictions(tmp_path: Path) -> None:
    seed_everything(1234)
    model = build_3d_cnn(Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="save-h5"))
    x, _ = _tiny_data(2, seed=42)
    predictions_before = model.predict(x, verbose=0)

    path = tmp_path / "model.h5"
    save_model(model, path)
    reloaded = load_model(path)
    predictions_after = reloaded.predict(x, verbose=0)

    np.testing.assert_allclose(predictions_before, predictions_after)
