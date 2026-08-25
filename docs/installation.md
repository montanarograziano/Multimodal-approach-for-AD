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
just install
```

`just install` (`uv sync --locked --all-groups --all-extras`) installs
every dependency group and optional extra against the locked versions in
`uv.lock`: runtime deps (nibabel, numpy, opencv-python, polars,
scikit-learn, scipy — everything `multimodal_ad.data` and the CLI import
unconditionally), the `dev` group (ruff, pyrefly, pytest, prek), the
`model` extra (TensorFlow, for `multimodal_ad.models`), the `science`
extra (`matplotlib`, `pillow`, for notebook-adjacent plotting), and the
`notebooks`/`docs` groups. This is the normal full local setup command; CI
uses narrower, targeted `uv sync`/`uv run` invocations per job to keep the
fast/model/notebook/docs jobs isolated (see `.github/workflows/`).

## Common commands

All commands are defined in the [`Justfile`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/Justfile):

| Command | Equivalent | Purpose |
| --- | --- | --- |
| `just install` | `uv sync --locked --all-groups --all-extras` | Install every dependency group and extra from the lockfile |
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
`notebooks` group, installed by `just install`:

```bash
just install
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

## VS Code / Pylance

Open this repository's root as the VS Code workspace, then select
`.venv/bin/python` as the interpreter (Command Palette → "Python: Select
 Interpreter"). `.vscode/settings.json` points Pylance at that interpreter
and at `src/` for module resolution. If Pylance reports false "missing
import" errors for `keras`/`tensorflow`/`nibabel`/etc., it is almost always
because `just install` has not been run yet (the `model`/`science` extras
are optional and only present after a full install) or a different
interpreter is still selected; re-run `just install` and re-select
`.venv/bin/python`. The project's authoritative type-checker is Pyrefly
(`just typecheck`/`just typecheck-model`); Pylance is IDE-only tooling, not
a CI gate.

## Verifying your setup

```bash
just check
```

should pass cleanly on a fresh clone before you start making changes.
