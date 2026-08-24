# Reproducibility

## What runs today

- `uv sync --locked` and `just check` (lint, format check, type check,
  `pytest`) run cleanly on a clean clone, no data required.
- `import multimodal_ad` works; the package exposes `__version__` and a
  docstring. That is its entire current surface area.
- This documentation site builds with `just docs-build`.

## What does not run today

- **None of the paper's scientific pipeline is ported.** The five legacy
  notebooks at the repository root (`Dataset_MRI.ipynb`,
  `Dataset_PET.ipynb`, `Training.ipynb`, `Heatmaps.ipynb`,
  `exploration.ipynb`) are the only place that logic exists, and per the
  [legacy notebook inventory](legacy-notebooks-inventory.md), **none of
  them run end-to-end outside the original author's interactive Colab
  session**: they use hardcoded placeholder paths
  (`DATA_DIR = Path("Your directory here")`), Colab-only imports
  (`google.colab.drive`), and reference variables from cells that are not
  present in the file as committed.
- There is no automated way, today, to reproduce the paper's reported
  metrics from this repository.

## Known blockers to full reproducibility

These are tracked in detail in the
[legacy notebook inventory](legacy-notebooks-inventory.md#summary-what-must-be-resolved-before-phase-2-scientific-code-porting):

1. Which central-frame depth (20, 30, or 50) produced the paper's reported
   numbers — all three appear across notebook cells.
2. Which of two conflicting `get_merged()` definitions (different fusion
   head shapes) was in effect.
3. Which of two conflicting `tuning_3d()` early-stopping `patience` values
   (35 vs. 20) was in effect.
4. Whether the "uncertain diagnosis → classify as demented" convention and
   the apparently garbled `other mental retarAD demion` regex fragment
   match the intended OASIS-3 diagnosis codebook.
5. Which cross-validation fold scheme (subject-level repeated 10×10 vs.
   frame-level single 10-fold) produced the reported metrics.
6. The AAL2 atlas file (`atlas.nii.gz`) referenced by `exploration.ipynb`
   is not in this repository and must be sourced separately.
7. MLflow experiment tracking was wired to an interactive, `input()`
   -prompted DagsHub project; it is not reproducible in CI or any
   non-interactive environment and needs a replacement.

Additionally, real OASIS-3 data cannot be committed or auto-fetched (see
[Data access & contracts](data-access.md)), so any reproduction attempt
requires the user's own OASIS-3 DUA and local data.

## Determinism notes (from the notebooks, not yet enforced in code)

- `RANDOM_SEED = 1234` is used consistently across dataset construction and
  cross-validation splits in the notebooks, but TensorFlow/Keras GPU
  training is not bitwise-deterministic across hardware/driver versions
  even with a fixed seed.
- Class balancing uses `random.sample()` for augmentation selection, which
  depends on Python's global `random` state, not a seeded local generator,
  in the current notebook code — a candidate fix during porting, not
  applied here.

## What "reproducible" will mean once Phase 2+ lands

A future PR should be able to update this page with: exact commands to
regenerate the dataset from a user-supplied OASIS-3 export, deterministic
seeding for both dataset construction and training, and either a
lightweight local experiment tracker or a documented, non-interactive
MLflow setup. None of that exists yet; this page will be updated alongside
that work rather than promising it in advance.
