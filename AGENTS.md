# AGENTS.md

Instructions for coding agents (and humans) working in this repository.

## Project

`multimodal-ad`: code for "Automated Detection of Alzheimer's Disease: A
Multi-modal Approach With 3D MRI and Amyloid PET"
([Scientific Reports, 2024](https://doi.org/10.1038/s41598-024-56001-9)).

The repository was migrated from a collection of exploratory Google Colab
notebooks (`Dataset_MRI.ipynb`, `Dataset_PET.ipynb`, `Training.ipynb`,
`Heatmaps.ipynb`, `exploration.ipynb`, plus `images/3D Brain Plot.ipynb`) to
a proper `src/` layout Python package (`multimodal_ad`). Those five
notebooks are preserved byte-for-byte under `notebooks/legacy/`; see
`docs/legacy-notebooks-inventory.md` for a full behavior audit before
touching any scientific logic. Small, newcomer-oriented thin notebooks
under `notebooks/` (not `notebooks/legacy/`) now call into the ported
`multimodal_ad` API on synthetic data; see "Notebooks" below.

**The typed data pipeline (`multimodal_ad.data`) has been ported**:
diagnosis normalization/labeling, a typed scan manifest, a local OASIS-3
layout adapter (no download/redistribution), volume loading/cropping/
resizing, deterministic augmentation, and subject-wise splitting.

**As of this PR, the model/training/evaluation/explainability pipeline
(`multimodal_ad.models`) has also been ported** from `Training.ipynb` and
`Heatmaps.ipynb`: the 3D CNN architecture and its frozen/named/fusion
variants, a training config/loop with deterministic seeding and `.keras`/
legacy `.h5` save-load, accuracy/sensitivity/specificity/AUC evaluation,
a native Grad-CAM, and AAL2 region-ranking utilities (`exploration.ipynb`).
`multimodal_ad.models` imports TensorFlow unconditionally; install the
`model` extra (`uv sync --extra model`) to use it. `multimodal_ad.data` and
the CLI do not depend on TensorFlow. See
`src/multimodal_ad/data/__init__.py` and `src/multimodal_ad/models/__init__.py`
for the module maps, and `docs/legacy-notebooks-inventory.md` for the
behavior audit each module's docstring cites.

## Environment

- Python: **3.13**, managed via `uv` (`.python-version` pins it).
- Package manager: **uv** (`pyproject.toml` + `uv.lock`). Do not use `pip`,
  `poetry`, `pipenv`, or `conda` in this project.
- Layout: `src/multimodal_ad/` (importable package), `tests/` (pytest),
  `docs/` (migration docs).

## Common commands

Run via `just` (see `Justfile` for the full list):

```sh
just install     # uv sync --locked --all-groups --all-extras
just fmt         # ruff format .
just lint        # ruff check .
just typecheck   # pyrefly check (src/tests, excludes multimodal_ad.models)
just typecheck-model  # pyrefly check src/multimodal_ad/models tests/models
just test        # pytest
just check       # lint + format check + typecheck + test (CI-equivalent)
just hooks       # uv run prek run --all-files
```

`just install` installs runtime, dev, model, science, notebook, and docs
dependencies in one shot (every dependency group and optional extra), for a
full local setup. CI jobs use narrower, targeted `uv sync`/`uv run` calls to
keep the fast/model/notebook/docs jobs isolated; see `.github/workflows/`.

Equivalent raw commands: `uv sync --locked --all-groups --all-extras`,
`uv run ruff check .`, `uv run ruff format --check .`, `uv run pyrefly check`,
`uv run pytest`, `uv run prek run --all-files`.

`multimodal_ad.models` (and `tests/models`) import TensorFlow/Keras
unconditionally, so they are excluded from `just typecheck`'s default scope
(`project-excludes` in `pyproject.toml`'s `[tool.pyrefly]`) to keep the fast
`check` CI job/`just check` runnable without the `model` extra installed.
Run `just typecheck-model` after `uv sync --extra model` to type-check that
subtree; CI's `model-smoke` job does this automatically after installing the
`model` extra.

## Conventions

- `[project.dependencies]` holds packages `multimodal_ad` unconditionally
  imports at runtime (currently: nibabel, numpy, opencv-python, pandas,
  scikit-learn, scipy, all used by `multimodal_ad.data`).
  `[project.optional-dependencies].science` holds packages only needed by
  code not yet ported or only used for specific features (TensorFlow/Keras,
  tf-keras-vis, matplotlib, pillow). When a module starts unconditionally
  importing a `science` package, move it to `[project.dependencies]`. Add
  dev tooling to `[dependency-groups].dev`. Notebook-only tooling
  (ipykernel, notebook, matplotlib for display) goes in
  `[dependency-groups].notebooks`.
- All `.ipynb` files are **excluded from Ruff lint/format** (see
  `extend-exclude` in `pyproject.toml`). `notebooks/legacy/*.ipynb` are
  preserved as historical artifacts, byte-for-byte; do not edit their cells
  without an explicit task to do so. `notebooks/*.ipynb` (the thin,
  ported-API notebooks) must stay orchestration-only: no function/class
  definitions, no Colab mounts/shell installs/absolute paths/credentials,
  committed with cleared outputs (`just notebooks-clear`); enforced by
  `tests/notebooks/` and the `notebook-smoke` CI job (`just
  notebooks-execute`).
- Type hints are required on new code in `src/`; `pyrefly check` runs in CI
  and via `prek`.
- Every non-trivial change needs a test in `tests/` (pytest). Prefer plain
  `assert`-based tests; no fixtures/frameworks beyond pytest unless the code
  under test needs them.
- Before committing, run `just check` (or let `prek` hooks run on commit).

## Migration plan pointer

Phase 0/1 (done): notebook inventory + project foundation (`pyproject.toml`,
`Justfile`, CI, lint/type/test scaffolding). Phase 2a (done): typed data
pipeline (`multimodal_ad.data`) ported from `Dataset_MRI.ipynb` /
`Dataset_PET.ipynb`, with a synthetic NIfTI/manifest generator so the full
data path is testable without OASIS-3. Phase 2b (done): model
training/evaluation/Grad-CAM/region-ranking (`multimodal_ad.models`) ported
from `Training.ipynb` / `Heatmaps.ipynb` / `exploration.ipynb`. Phase 3
(this PR, done): thin notebooks under `notebooks/` calling into the ported
API on synthetic data; the five original notebooks (plus `images/3D Brain
Plot.ipynb`) moved byte-for-byte to `notebooks/legacy/`. See
`docs/legacy-notebooks-inventory.md` for the open scientific questions
(frame depth, fusion head, CV fold scheme, etc.) that remain unresolved and
block real-OASIS-3 validation, not this port; several data-path ambiguities
are also documented inline in `multimodal_ad.data`/`multimodal_ad.models`
module docstrings where an explicit, documented choice had to be made.
