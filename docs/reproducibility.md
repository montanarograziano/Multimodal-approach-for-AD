# Reproducibility

## What runs today

- `uv sync --locked` and `just check` (lint, format check, `pyrefly
  check`, `pytest`) run cleanly on a clean clone, CPU-only, no data
  required. This exercises `multimodal_ad.data` and the CLI; it does not
  need TensorFlow.
- `uv sync --locked --extra model`, `just typecheck-model`, and `uv run
  pytest` (now including `tests/models/`) also run cleanly, CPU-only.
  This exercises `multimodal_ad.models` (3D CNN, training loop,
  evaluation, Grad-CAM, region ranking) against tiny synthetic data and
  a handful of training steps, not full training runs.
- `python -m multimodal_ad.cli` (the synthetic quickstart) generates a
  small deterministic fake dataset and runs it through manifest
  construction, subject-wise splitting, and volume preprocessing, in
  seconds on a laptop CPU.
- This documentation site builds with `just docs-build`.

**None of this reproduces the paper's published metrics or validates the
pipeline against real OASIS-3 data.** Every test and the CLI quickstart
run against synthetic, generated-on-the-fly volumes/labels
(`multimodal_ad.data.synthetic`), which have the right shape/dtype/session-id
contract but no clinical meaning.

## What "ported" means here, precisely

`multimodal_ad.data` and `multimodal_ad.models` implement the
scientifically meaningful logic from the five legacy notebooks
(`Dataset_MRI.ipynb`, `Dataset_PET.ipynb`, `Training.ipynb`,
`Heatmaps.ipynb`, `exploration.ipynb`), each as typed, tested, importable
Python. But several implementation details the notebooks left ambiguous
or contradictory across cells are resolved here as an **explicit,
documented default**, not a confirmed match to whatever configuration
actually produced the paper's reported numbers. Every such choice is
recorded in the relevant module's docstring; see
[Methodology](methodology.md#ported-with-documented-divergences-and-open-ambiguities)
for a summary and the
[legacy notebook inventory](legacy-notebooks-inventory.md) for the full
audit.

The legacy `.ipynb` files themselves are **unmodified** and still not
runnable end-to-end (hardcoded placeholder paths, Colab-only imports,
undefined names across cells) — they remain historical reference
material, not something you can execute.

## Known blockers to full reproducibility

These are tracked in detail in the
[legacy notebook inventory](legacy-notebooks-inventory.md#summary-what-must-be-resolved-before-phase-2-scientific-code-porting):

1. Which central-frame depth (20, 30, or 50) produced the paper's reported
   numbers — all three appear across notebook cells. `data.volumes`
   defaults to 50, exposed as a config field.
2. Which of two conflicting `get_merged()` definitions (different fusion
   head shapes) was in effect. `models.architecture.build_fusion_model`
   implements the non-buggy one (`Dense(128, relu)` head).
3. Which of two conflicting `tuning_3d()` early-stopping `patience` values
   (35 vs. 20) was in effect. `models.training.TrainingConfig` defaults to
   35.
4. Whether the "uncertain diagnosis → classify as demented" convention and
   the apparently garbled `other mental retarAD demion` regex fragment
   match the intended OASIS-3 diagnosis codebook. `data.labeling` retains
   the notebook's behavior as-is, flagged as unconfirmed.
5. Which cross-validation fold scheme (subject-level repeated 10×10 vs.
   frame-level single 10-fold) produced the reported metrics. Neither is
   wired into a CV driver here; `data.splits` provides subject-wise
   splitting/stratified folds as building blocks only.
6. The AAL2 atlas file (`atlas.nii.gz`) referenced by `exploration.ipynb`
   is not in this repository and must be sourced separately;
   `models.regions` requires callers to load it themselves.
7. MLflow experiment tracking was wired to an interactive, `input()`
   -prompted DagsHub project; `models.training` drops it entirely rather
   than replacing it (see that module's docstring).

Additionally, real OASIS-3 data cannot be committed or auto-fetched (see
[Data access & contracts](data-access.md)), so any real validation attempt
requires the user's own OASIS-3 DUA and local data, run through
`multimodal_ad.data.oasis` (whose CSV column contract is a best-effort
inference from the notebooks, not yet confirmed against a real export).

## Determinism notes

- `RANDOM_SEED = 1234` from the notebooks is retained as `DEFAULT_SEED`
  across `data.synthetic`, `models.training.seed_everything` (Python,
  NumPy, and TensorFlow RNGs), and test fixtures.
- `models.training.seed_everything` gives bitwise-deterministic runs on
  CPU for the ops this package uses; TensorFlow does not guarantee
  bitwise determinism on GPU even with a fixed seed.
- `data.augmentation.augment_volume` takes an explicit
  `numpy.random.Generator` instead of the notebooks' process-global
  `random` module, so augmentation is reproducible independent of import
  order.

## What would make this a real reproduction

A future round of work would need to: (1) get authors' input on the open
ambiguities above, (2) run the ported pipeline end-to-end against a real
OASIS-3 export under a proper DUA, and (3) compare the resulting metrics
against the paper's Table results. None of that has happened; this page
will be updated if and when it does, rather than promising it in advance.
