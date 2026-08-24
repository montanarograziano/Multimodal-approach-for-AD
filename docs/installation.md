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
notebook-adjacent plotting; no `multimodal_ad` runtime code imports it.

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
