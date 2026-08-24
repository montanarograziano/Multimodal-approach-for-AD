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
pytest, prek) against the locked versions in `uv.lock`. It does **not**
install the heavy scientific stack (TensorFlow, OpenCV, nibabel, ...) by
default, because `multimodal_ad` does not import them yet:

```bash
uv sync --locked --extra science
```

## Common commands

All commands are defined in the [`Justfile`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/Justfile):

| Command | Equivalent | Purpose |
| --- | --- | --- |
| `just sync` | `uv sync --locked` | Install dependencies from the lockfile |
| `just fmt` | `uv run ruff format .` | Format code |
| `just lint` | `uv run ruff check .` | Lint code |
| `just typecheck` | `uv run pyrefly check` | Type-check `src/` and `tests/` |
| `just test` | `uv run pytest` | Run the test suite |
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
