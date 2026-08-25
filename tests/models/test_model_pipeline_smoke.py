"""End-to-end smoke test: build -> train -> evaluate -> save/reload -> Grad-CAM.

No OASIS-3 or pretrained weights needed: everything runs on tiny synthetic
arrays, on CPU, to keep this fast enough for CI (`model-smoke` job in
`.github/workflows/ci.yml`).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from multimodal_ad.models.architecture import Cnn3DConfig, build_3d_cnn  # noqa: E402
from multimodal_ad.models.evaluation import evaluate_predictions  # noqa: E402
from multimodal_ad.models.gradcam import (  # noqa: E402
    last_conv_layer_name,
    make_gradcam_heatmap,
    unwrap_output_activation,
)
from multimodal_ad.models.training import TrainingConfig, save_model, train_model  # noqa: E402

_TINY = 48


def test_full_model_pipeline_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    x_train = rng.random((6, _TINY, _TINY, _TINY, 1)).astype("float32")
    y_train = (np.arange(6) % 2).astype("float32")
    x_val = rng.random((4, _TINY, _TINY, _TINY, 1)).astype("float32")
    y_val = (np.arange(4) % 2).astype("float32")

    model = build_3d_cnn(Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="smoke"))
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        checkpoint_path=tmp_path / "smoke.keras",
        early_stopping_baseline=None,
        verbose=0,
    )
    result = train_model(model, x_train, y_train, x_val, y_val, config)
    assert np.isfinite(result.val_loss)

    probabilities = result.model.predict(x_val, verbose=0).reshape(-1)
    metrics = evaluate_predictions(y_val, probabilities)
    assert 0.0 <= metrics.accuracy <= 1.0

    save_path = tmp_path / "smoke-final.keras"
    save_model(result.model, save_path)
    assert save_path.exists()

    gradcam_config = Cnn3DConfig(width=_TINY, height=_TINY, depth=_TINY, name="smoke-cam")
    gradcam_model = build_3d_cnn(gradcam_config)
    unwrap_output_activation(gradcam_model)
    layer_name = last_conv_layer_name(gradcam_model)
    heatmap = make_gradcam_heatmap(x_val[:1], gradcam_model, layer_name)
    assert np.all(np.isfinite(heatmap))
    assert np.all((heatmap >= 0.0) & (heatmap <= 1.0))
