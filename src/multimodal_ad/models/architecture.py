"""3D CNN model builders ported from `Training.ipynb` (see
`docs/legacy-notebooks-inventory.md`, "`Training.ipynb`" section).

The notebook defines four near-identical Keras functional models
(`get_3d_model`, `get_3d_frozen`, `get_3d_x`, `get_3d_y`) that differ only in
which layers are frozen (`trainable=False`) and whether the `Input`/
`Dropout`/output layers carry explicit names (needed later to build the
fusion model). This module collapses them into one parameterized builder,
`build_3d_cnn`, plus thin named wrappers so call sites read like the
notebook's intent.

**Documented topology** (retained exactly, all four notebook variants):
`Input(width, height, depth, 1)` -> four `(Conv3D(kernel=3, relu) ->
MaxPool3D(pool=2) -> BatchNormalization)` blocks with filters
`(64, 64, 128, 256)` -> `GlobalAveragePooling3D` -> `Dense(512, relu)` ->
`Dropout(0.3)` -> `Dense(1, sigmoid)`. All `Conv3D`/`MaxPool3D` use Keras's
default `padding="valid"` (never set explicitly in the notebook).

**Documented, unresolved freeze inconsistency**: `get_3d_frozen` freezes only
`Conv3D`/`BatchNormalization` layers (the `Dense(512)` head stays
trainable), while `get_3d_x`/`get_3d_y` (the variants used to build the
fusion model) *also* freeze `Dense(512)`. The notebook never reconciles
this; both behaviors are preserved here via the independent `freeze_conv`
and `freeze_dense` flags rather than guessing which was "correct" for any
one reported experiment.

**Documented fusion-head ambiguity**: `Training.ipynb` defines `get_merged()`
twice. The first definition concatenates each modality's `Dense(512)`
feature layer (`model.layers[-3].output`) and adds
`Dense(128, relu) -> Dropout(0.3) -> Dense(1, sigmoid)`. The second
definition (which shadows the first in a live notebook session) reassigns
`z = Dropout(0.3)(merged)` immediately after computing an unused
`Dense(4, relu)`, an apparent copy-paste bug that discards the `Dense(4,
...)` layer entirely. `build_fusion_model` below implements the first,
non-buggy definition (`Dense(128, relu)` head) as the canonical fusion
architecture; which definition actually produced the paper's reported
merged-model numbers cannot be determined from static inspection and is an
open question for the paper authors (see inventory doc, open question 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import layers

#: Retained from the legacy notebooks (`RANDOM_SEED = 1234`); filter widths
#: for the four Conv3D blocks, per the Zunair et al. architecture
#: (https://arxiv.org/abs/2007.13224) cited in the notebook.
DEFAULT_FILTERS: tuple[int, int, int, int] = (64, 64, 128, 256)
DEFAULT_KERNEL_SIZE = 3
DEFAULT_POOL_SIZE = 2
DEFAULT_DENSE_UNITS = 512
DEFAULT_DROPOUT = 0.3


@dataclass(frozen=True)
class Cnn3DConfig:
    """Shape/topology knobs for `build_3d_cnn`.

    Defaults reproduce the legacy `(128, 128, 50, 1)` input and
    `(64, 64, 128, 256)` filter topology exactly; only shape/filter/epoch
    values are meant to be overridden (e.g. for fast CPU tests), not the
    layer ordering itself.
    """

    width: int = 128
    height: int = 128
    depth: int = 50
    filters: tuple[int, ...] = DEFAULT_FILTERS
    kernel_size: int = DEFAULT_KERNEL_SIZE
    pool_size: int = DEFAULT_POOL_SIZE
    dense_units: int = DEFAULT_DENSE_UNITS
    dropout: float = DEFAULT_DROPOUT
    freeze_conv: bool = False
    freeze_dense: bool = False
    output_activation: str = "sigmoid"
    input_name: str | None = None
    dropout_name: str | None = None
    output_name: str | None = None
    name: str = "3d-cnn"


def build_3d_cnn(config: Cnn3DConfig | None = None) -> keras.Model:
    """Build the notebook's 3D CNN, parameterized by `config`.

    `config.freeze_conv` / `config.freeze_dense` select between the four
    notebook variants (see module docstring): both `False` is
    `get_3d_model`; `freeze_conv=True, freeze_dense=False` is
    `get_3d_frozen`; both `True` (plus `input_name`/`dropout_name`) is
    `get_3d_x`/`get_3d_y`.
    """
    config = config if config is not None else Cnn3DConfig()
    inputs = keras.Input((config.width, config.height, config.depth, 1), name=config.input_name)

    x = inputs
    for filters in config.filters:
        x = layers.Conv3D(
            filters=filters,
            kernel_size=config.kernel_size,
            activation="relu",
            trainable=not config.freeze_conv,
        )(x)
        x = layers.MaxPool3D(pool_size=config.pool_size)(x)
        x = layers.BatchNormalization(trainable=not config.freeze_conv)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(
        units=config.dense_units, activation="relu", trainable=not config.freeze_dense
    )(x)
    x = layers.Dropout(config.dropout, name=config.dropout_name)(x)
    outputs = layers.Dense(units=1, activation=config.output_activation, name=config.output_name)(x)

    return keras.Model(inputs, outputs, name=config.name)


def build_mri_input_model(config: Cnn3DConfig, *, name: str = "mri") -> keras.Model:
    """`get_3d_x` equivalent: a fusion-ready branch named `input_x`/`drop_x`."""
    branch_config = _replace(
        config,
        input_name="input_x",
        dropout_name="drop_x",
        freeze_conv=True,
        freeze_dense=True,
        name=name,
    )
    return build_3d_cnn(branch_config)


def build_pet_input_model(config: Cnn3DConfig, *, name: str = "pet") -> keras.Model:
    """`get_3d_y` equivalent: a fusion-ready branch named `input_y`/`drop_y`/`out_y`."""
    branch_config = _replace(
        config,
        input_name="input_y",
        dropout_name="drop_y",
        output_name="out_y",
        freeze_conv=True,
        freeze_dense=True,
        name=name,
    )
    return build_3d_cnn(branch_config)


def _replace(config: Cnn3DConfig, **overrides: object) -> Cnn3DConfig:
    from dataclasses import replace

    return replace(config, **overrides)  # type: ignore[arg-type]


def extract_feature_extractor(model: keras.Model) -> keras.Model:
    """Return a model whose output is `model`'s penultimate `Dense` feature layer.

    Matches the legacy `Model(inputs=model.input, outputs=model.layers[-3].output)`
    slicing used by `get_merged()` to peel off the `Dropout`/output `Dense`
    layers and expose the `Dense(dense_units, relu)` feature vector.
    """
    return keras.Model(
        inputs=model.input, outputs=model.layers[-3].output, name=f"{model.name}-features"
    )


def build_fusion_model(
    mri_model: keras.Model,
    pet_model: keras.Model,
    *,
    hidden_units: int = 128,
    dropout: float = DEFAULT_DROPOUT,
    output_activation: str = "sigmoid",
    name: str = "merged",
) -> keras.Model:
    """`get_merged()` (canonical, non-buggy definition; see module docstring).

    `mri_model`/`pet_model` are typically produced by `build_mri_input_model`/
    `build_pet_input_model` with pretrained weights already loaded (transfer
    learning), matching the notebook's `mri_model.load_weights(...)` /
    `pet_model.load_weights(...)` calls before merging.
    """
    mri_features = extract_feature_extractor(mri_model)
    pet_features = extract_feature_extractor(pet_model)
    merged = layers.concatenate([mri_features.output, pet_features.output])
    z = layers.Dense(hidden_units, activation="relu")(merged)
    z = layers.Dropout(dropout)(z)
    outputs = layers.Dense(1, activation=output_activation)(z)
    return keras.Model(inputs=[mri_features.input, pet_features.input], outputs=outputs, name=name)
