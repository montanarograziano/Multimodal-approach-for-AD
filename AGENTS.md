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

**As of this PR, no scientific code has been ported yet.** The package is a
scaffold. Do not assume `multimodal_ad` implements any of the notebooks'
pipelines until a phase explicitly ports them.

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
just hooks       # prek run --all-files
```

Equivalent raw commands: `uv sync --locked`, `uv run ruff check .`,
`uv run ruff format --check .`, `uv run pyrefly check`, `uv run pytest`.

## Conventions

- Add production dependencies to `[project.optional-dependencies].science`
  (heavy scientific stack: TensorFlow/Keras, NumPy, OpenCV, nibabel, etc.),
  not to `[project.dependencies]`, until the package actually imports them.
  Add dev tooling to `[dependency-groups].dev`. Notebook-only tooling
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

Phase 0/1 (this PR): notebook inventory + project foundation (this file,
`pyproject.toml`, `Justfile`, CI, lint/type/test scaffolding). Scientific
code porting (Phase 2+) is a separate, stacked PR; see
`docs/legacy-notebooks-inventory.md` for the open questions that must be
resolved with the paper authors before that work starts.
