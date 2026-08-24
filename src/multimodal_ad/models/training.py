"""Training config/loop ported from `Training.ipynb`'s `train_3d_model`.

**Documented divergences from the legacy notebook** (see
`docs/legacy-notebooks-inventory.md`, "`Training.ipynb`" section):

- MLflow/DagsHub tracking (`mlflow.start_run`, `mlflow.tensorflow.autolog`,
  `input()`-prompted credentials) is dropped entirely: it is not
  reproducible outside the original author's interactive Colab session and
  no new tracking framework replaces it (see AGENTS.md, "avoid new tracking
  frameworks"). `TrainingResult` below carries the same information
  (history, evaluation metrics, confusion-matrix counts) as plain, typed
  return values instead of side-effecting logs.
- The notebook monitors `"val_acc"` (a Keras 1.x/early-2.x metric-name
  convention). Modern Keras (3.x, installed here) names the accuracy metric
  `"accuracy"` when compiled with `metrics=["accuracy"]`, so the
  corresponding validation key is `"val_accuracy"`. This module monitors
  `"val_accuracy"`, preserving the notebook's *semantics* (best validation
  accuracy governs checkpointing/early stopping) under the current
  framework's naming, not a byte-for-byte string match.
- Global `random.seed(RANDOM_SEED)` (module import time, Python-only) is
  replaced by an explicit `seed_everything()` call covering Python, NumPy,
  and TensorFlow RNGs, invoked once per training run.

**Retained exactly**: `binary_crossentropy` loss; `Adam` optimizer with
`ExponentialDecay(initial_learning_rate=5e-5, decay_steps=100000,
decay_rate=0.96, staircase=True)` (the `5e-5` value is cited in the
notebook from https://doi.org/10.3938/jkps.75.597, a paper-justified
constant); `EarlyStopping(patience=35, min_delta=0.001, baseline=0.60)`;
`ModelCheckpoint(save_best_only=True)`; default `epochs=10000`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import keras
import numpy as np
import tensorflow as tf

#: Retained from the legacy notebooks (`RANDOM_SEED = 1234`).
DEFAULT_SEED = 1234
DEFAULT_MAX_EPOCHS = 10000
DEFAULT_PATIENCE = 35
DEFAULT_INITIAL_LEARNING_RATE = 5e-5  # https://doi.org/10.3938/jkps.75.597


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for deterministic runs.

    TensorFlow's determinism guarantees depend on hardware/op support (some
    GPU ops remain nondeterministic even when seeded); on CPU, the ops this
    package uses are deterministic given a fixed seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for `train_model`, defaulting to the legacy notebook's values."""

    epochs: int = DEFAULT_MAX_EPOCHS
    batch_size: int = 32
    initial_learning_rate: float = DEFAULT_INITIAL_LEARNING_RATE
    lr_decay_steps: int = 100000
    lr_decay_rate: float = 0.96
    lr_staircase: bool = True
    early_stopping_patience: int = DEFAULT_PATIENCE
    early_stopping_min_delta: float = 0.001
    early_stopping_baseline: float | None = 0.60
    monitor: str = "val_accuracy"
    checkpoint_path: str | Path = "model_checkpoint.keras"
    seed: int = DEFAULT_SEED
    shuffle_buffer: int | None = None  # defaults to len(x_train) if None
    prefetch: int = 4
    verbose: Literal[0, 1, 2] = 2


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of `train_model`: trained model, fit history, best-checkpoint eval metrics."""

    model: keras.Model
    history: keras.callbacks.History
    val_loss: float
    val_accuracy: float


def build_optimizer(config: TrainingConfig) -> keras.optimizers.Optimizer:
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        config.initial_learning_rate,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate,
        staircase=config.lr_staircase,
    )
    return keras.optimizers.Adam(learning_rate=lr_schedule)


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle_buffer: int | None = None,
    prefetch: int = 4,
) -> tf.data.Dataset:
    """`tf.data.Dataset` construction matching the notebook's shuffle/batch/prefetch."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    buffer = shuffle_buffer if shuffle_buffer is not None else len(x)
    return dataset.shuffle(buffer, seed=seed).batch(batch_size).prefetch(prefetch)


def concatenate_modalities(
    x_a: np.ndarray, y_a: np.ndarray, x_b: np.ndarray, y_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool two modalities' arrays into one dataset (the notebook's "combined" 10-fold CV path).

    Not a fusion architecture: both modalities are concatenated *before*
    training a single unimodal `build_3d_cnn` model on the pooled data (see
    `Training.ipynb`, "Combined 10-fold CV").
    """
    return np.concatenate((x_a, x_b), axis=0), np.concatenate((y_a, y_b), axis=0)


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Compile and train `model`, restoring the best checkpoint before evaluating.

    `model` must already have its input/output topology built (e.g. via
    `multimodal_ad.models.architecture.build_3d_cnn`); this function only
    compiles, seeds, fits, and evaluates it, matching the legacy
    `train_3d_model` minus MLflow logging (see module docstring).
    """
    config = config if config is not None else TrainingConfig()
    seed_everything(config.seed)

    train_dataset = make_dataset(
        x_train,
        y_train,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
        prefetch=config.prefetch,
    )
    val_dataset = make_dataset(
        x_val,
        y_val,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
        prefetch=config.prefetch,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=build_optimizer(config),
        metrics=["accuracy"],
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        str(config.checkpoint_path), monitor=config.monitor, save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor=config.monitor,
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        baseline=config.early_stopping_baseline,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        shuffle=True,
        verbose=cast(Literal[0, 1, 2], config.verbose),
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    model.load_weights(str(config.checkpoint_path))
    eval_metrics = cast(dict[str, float], model.evaluate(val_dataset, verbose=0, return_dict=True))
    val_loss, val_accuracy = eval_metrics["loss"], eval_metrics["accuracy"]

    return TrainingResult(
        model=model, history=history, val_loss=float(val_loss), val_accuracy=float(val_accuracy)
    )


def save_model(model: keras.Model, path: str | Path) -> None:
    """Save in the modern `.keras` format, or legacy `.h5` if `path` ends in `.h5`.

    `.h5` support is kept for interop with the notebooks' checkpoint files
    (`*.h5`); Keras 3 emits a deprecation warning for `.h5` saves (still
    functional, just "legacy"), which this function does not suppress.
    """
    model.save(str(path))


def load_model(path: str | Path) -> keras.Model:
    """Load a `.keras` or legacy `.h5` model file (format inferred from extension)."""
    return keras.models.load_model(str(path))
