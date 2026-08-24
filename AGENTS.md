# AGENTS.md

Instructions for coding agents (and humans) working in this repository.

## Project

`multimodal-ad`: code for "Automated Detection of Alzheimer's Disease: A
Multi-modal Approach With 3D MRI and Amyloid PET"
([Scientific Reports, 2024](https://doi.org/10.1038/s41598-024-56001-9)).

The repository is mid-migration from a collection of exploratory Google
Colab notebooks (`Dataset_MRI.ipynb`, `Dataset_PET.ipynb`, `Training.ipynb`,
`Heatmaps.ipynb`, `exploration.ipynb`) to a proper `src/` layout Python
package (`multimodal_ad`). See `docs/legacy-notebooks-inventory.md` for a
full behavior audit of the legacy notebooks before touching any scientific
logic.

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
just sync        # uv sync --locked
just fmt         # ruff format .
just lint        # ruff check .
just typecheck   # pyrefly check
just test        # pytest
just check       # lint + format check + typecheck + test (CI-equivalent)
just hooks       # uv run prek run --all-files
```

Equivalent raw commands: `uv sync --locked`, `uv run ruff check .`,
`uv run ruff format --check .`, `uv run pyrefly check`, `uv run pytest`,
`uv run prek run --all-files`.

## Conventions

- `[project.dependencies]` holds packages `multimodal_ad` unconditionally
  imports at runtime (currently: nibabel, numpy, opencv-python, pandas,
  scikit-learn, scipy, all used by `multimodal_ad.data`).
  `[project.optional-dependencies].science` holds packages only needed by
  code not yet ported or only used for specific features (TensorFlow/Keras,
  tf-keras-vis, matplotlib, pillow). When a module starts unconditionally
  importing a `science` package, move it to `[project.dependencies]`. Add
  dev tooling to `[dependency-groups].dev`. Notebook-only tooling
  (ipykernel, notebook) goes in `[dependency-groups].notebooks`.
- Legacy `.ipynb` files are **excluded from Ruff lint/format** (see
  `extend-exclude` in `pyproject.toml`) and are preserved as historical
  artifacts. Do not edit notebook cells without an explicit task to do so;
  when you do, keep the diff minimal and note the change in the PR/commit.
- Type hints are required on new code in `src/`; `pyrefly check` runs in CI
  and via `prek`.
- Every non-trivial change needs a test in `tests/` (pytest). Prefer plain
  `assert`-based tests; no fixtures/frameworks beyond pytest unless the code
  under test needs them.
- Before committing, run `just check` (or let `prek` hooks run on commit).

## Migration plan pointer

Phase 0/1: notebook inventory + project foundation (`pyproject.toml`,
`Justfile`, CI, lint/type/test scaffolding). Phase 2a (this PR): typed data
pipeline (`multimodal_ad.data`) ported from `Dataset_MRI.ipynb` /
`Dataset_PET.ipynb`, with a synthetic NIfTI/manifest generator so the full
data path is testable without OASIS-3. Phase 2b (future, stacked): model
training/inference ported from `Training.ipynb` / `Heatmaps.ipynb` /
`exploration.ipynb`. See `docs/legacy-notebooks-inventory.md` for the open
questions (frame depth, fusion head, CV fold scheme, etc.) that must be
resolved with the paper authors before Phase 2b starts; several data-path
ambiguities are also documented inline in `multimodal_ad.data` module
docstrings where this PR had to make an explicit, documented choice.
