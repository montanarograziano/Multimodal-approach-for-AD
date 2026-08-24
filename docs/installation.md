# Installation

## Requirements

- Python **3.13** (pinned in `.python-version`).
- [`uv`](https://docs.astral.sh/uv/) as the package manager. Do not use
  `pip`, `poetry`, `pipenv`, or `conda` for the root project.
- [`just`](https://github.com/casey/just) (optional, but every command
  below has a `just` recipe).

## Setup

```bash
git clone https://github.com/montanarograziano/Multimodal-approach-for-AD.git
cd Multimodal-approach-for-AD
uv sync --locked
```

`uv sync --locked` installs the `dev` dependency group (ruff, pyrefly,
pytest, prek) plus `multimodal_ad`'s always-on runtime dependencies
(nibabel, numpy, opencv-python, pandas, scikit-learn, scipy — everything
`multimodal_ad.data` and the CLI import unconditionally), against the
locked versions in `uv.lock`. It does **not** install TensorFlow by
default, because only `multimodal_ad.models` imports it:

```bash
uv sync --locked --extra model     # + TensorFlow, for multimodal_ad.models
```

There's also a `science` extra (`matplotlib`, `pillow`) for
notebook-adjacent plotting; no `multimodal_ad` runtime code imports it. To
run the notebooks under `notebooks/` (see [Notebooks](#notebooks) below),
install the `notebooks` dependency group too:

```bash
uv sync --locked --extra model --group notebooks   # notebooks + model pipeline
```

## Common commands

All commands are defined in the [`Justfile`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/Justfile):

| Command | Equivalent | Purpose |
| --- | --- | --- |
| `just sync` | `uv sync --locked` | Install dependencies from the lockfile |
| `just fmt` | `uv run ruff format .` | Format code |
| `just lint` | `uv run ruff check .` | Lint code |
| `just typecheck` | `uv run pyrefly check` | Type-check `src/`/`tests/`, excluding `multimodal_ad.models`/`tests/models` |
| `just typecheck-model` | `uv run pyrefly check src/multimodal_ad/models tests/models` | Type-check the model subtree (needs `--extra model`) |
| `just test` | `uv run pytest` | Run the test suite (model tests auto-skip without `--extra model`) |
| `just check` | lint + format check + typecheck + test | CI-equivalent local check |
| `just hooks` | `uv run prek run --all-files` | Run all pre-commit hooks |
| `just docs-serve` | `uv run --group docs zensical serve` | Live-reload docs preview |
| `just docs-build` | `uv run --group docs zensical build --strict` | Build the static docs site |
| `just notebooks-launch` | `uv run --group notebooks jupyter notebook notebooks/` | Launch Jupyter against `notebooks/` |
| `just notebooks-execute` | `uv run --group notebooks jupyter nbconvert --to notebook --execute ...` | Execute every thin notebook from a clean kernel (CI-equivalent) |
| `just notebooks-clear` | `uv run --group notebooks jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` | Clear notebook outputs before committing |

## Notebooks

`notebooks/*.ipynb` (not `notebooks/legacy/`) are small, newcomer-oriented
notebooks that call into the `multimodal_ad` API on deterministic synthetic
data: no OASIS-3 access, no manual paths, no Colab. `01-data-quickstart.ipynb`
needs only the base install; `02-tiny-model-workflow.ipynb` and
`03-explainability.ipynb` need the `model` extra too. All three need the
`notebooks` group:

```bash
uv sync --locked --extra model --group notebooks
just notebooks-launch
```

Commit notebooks with cleared outputs (`just notebooks-clear`); CI enforces
this and re-executes every notebook from a clean kernel (`notebook-smoke`
job). `notebooks/legacy/*.ipynb` are the original Colab notebooks, preserved
byte-for-byte and not executable outside their original environment (see
[legacy notebook inventory](legacy-notebooks-inventory.md)).

## Git hooks

```bash
uv run prek install
```

installs the hooks from `.pre-commit-config.yaml` (ruff, whitespace/EOF
checks, TOML/YAML validation, large-file guard, pyrefly) to run on every
commit.

## Verifying your setup

```bash
just check
```

should pass cleanly on a fresh clone before you start making changes.
