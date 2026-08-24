"""Native 3D Grad-CAM, ported from `Heatmaps.ipynb`.

**Documented choice**: `Heatmaps.ipynb` computes Grad-CAM two ways: a
hand-written, unused 2D `make_gradcam_heatmap()` (standard single-layer
Keras Grad-CAM recipe, adapted from
https://keras.io/examples/vision/grad_cam/), and the actually-used
[`tf-keras-vis`](https://github.com/keisen/tf-keras-vis) `Gradcam` class
with `penultimate_layer=-7` (fragile: a numeric layer offset, not a name,
and coincidentally equal to the batch chunk size used in the notebook's
loop, per the inventory doc). This module ports the *unused* hand-written
recipe instead, generalized from 2D to 3D and from a numeric layer offset
to an explicit `layer_name`, per AGENTS.md's instruction to "prefer
TensorFlow/Keras primitives over tf-keras-vis if it removes compatibility
risk" and the inventory doc's flag that hardcoded numeric layer indices are
"fragile Keras auto-naming ... must be replaced with explicit `name=`
layer arguments when ported." `tf-keras-vis` is no longer a dependency
(see `pyproject.toml`).

**Retained**: unwrapping the output activation before computing gradients
(`model.layers[-1].activation = None` in the notebook) so gradients are
computed against raw logits, not squashed sigmoid outputs; per-channel mean
pooling of gradients (`tf.reduce_mean(grads, axis=(0, 1, 2))` in 2D becomes
`axis=(0, 1, 2, 3)` for a 3D `(D, H, W, C)` feature map); ReLU + max
normalization to `[0, 1]` (`tf.maximum(heatmap, 0) / reduce_max(heatmap)`).
`tf.config.run_functions_eagerly(True)` is not required by this
`tf.GradientTape`-based port (unlike the notebook's `tf-keras-vis` path,
which needed it to compute gradients against a graph-mode model); Keras 3's
eager-by-default execution makes it unnecessary here.

**Retained, target-layer selection**: the notebook's `layer_name` picks the
last `Conv3D` layer (`'conv3d_3'`, the 4th and final convolutional block,
by 0-indexed auto-naming). This module requires callers to pass an explicit
`layer_name` (stable, not autoname-order-dependent); `last_conv_layer_name`
below finds it by type for convenience.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


def last_conv_layer_name(model: keras.Model, *, layer_type: type = keras.layers.Conv3D) -> str:
    """Return the name of the last layer of `layer_type` in `model` (default: `Conv3D`).

    Raises `ValueError` if no matching layer exists.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layer_type):
            return layer.name
    raise ValueError(f"model {model.name!r} has no layer of type {layer_type.__name__}")


def make_gradcam_heatmap(
    volume: np.ndarray,
    model: keras.Model,
    layer_name: str,
    *,
    pred_index: int | None = None,
) -> np.ndarray:
    """Compute a 3D Grad-CAM heatmap for a single volume against `layer_name`.

    `volume` is a single sample with a leading batch dimension already
    present (shape `(1, D, H, W, 1)`, matching the model's `Input` shape).
    Returns a heatmap shaped like `layer_name`'s spatial output (typically
    smaller than the input volume, since Grad-CAM operates on a
    late/coarse feature map), finite and normalized to `[0, 1]`.
    """
    grad_model = keras.Model(
        inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(volume)
        if pred_index is None:
            pred_index = 0  # binary sigmoid head: single output unit
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    # Mean gradient per channel over the spatial (D, H, W) axes of a
    # (batch, D, H, W, C) feature map.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    last_conv_layer_output = last_conv_layer_output[0]
    pooled_grads_column = tf.expand_dims(pooled_grads, axis=-1)
    heatmap = last_conv_layer_output @ pooled_grads_column
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.math.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, max_value)
    return heatmap.numpy()


def unwrap_output_activation(model: keras.Model) -> None:
    """Strip the output layer's activation in place (Grad-CAM needs raw logits).

    Mutates `model.layers[-1].activation`, matching the notebook's
    `model.layers[-1].activation = None`. Callers must rebuild/recompile
    downstream computations against the mutated model; this does not clone
    the model.
    """
    output_layer = model.layers[-1]
    output_layer.activation = keras.activations.linear
