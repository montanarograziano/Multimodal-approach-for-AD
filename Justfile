set shell := ["bash", "-cu"]

# List available recipes
default:
    @just --list

# Install project dependencies (dev group) from the lock file
sync:
    uv sync --locked

# Format code with Ruff
fmt:
    uv run ruff format .

# Lint code with Ruff
lint:
    uv run ruff check .

# Type-check with Pyrefly (non-model source/tests only; no TensorFlow needed)
typecheck:
    uv run pyrefly check

# Type-check multimodal_ad.models + tests/models (requires `uv sync --extra model`)
typecheck-model:
    uv run pyrefly check src/multimodal_ad/models tests/models

# Run the test suite
test:
    uv run pytest

# Run everything CI checks: lint, format check, type-check, tests
check: lint
    uv run ruff format --check .
    just typecheck
    just test

# Run all prek hooks against the whole repo
hooks:
    uv run prek run --all-files

# Serve the documentation site locally with live reload
docs-serve:
    uv run --group docs zensical serve

# Build the documentation site, failing on warnings
docs-build:
    uv run --group docs zensical build --strict

# Launch Jupyter Notebook against notebooks/ (needs `--extra model` for 02/03)
notebooks-launch:
    uv run --group notebooks jupyter notebook notebooks/

# Execute every thin notebook from a clean kernel, failing on any cell error
# (needs `uv sync --extra model --group notebooks`; excludes notebooks/legacy/,
# which are preserved, non-executable historical artifacts)
notebooks-execute:
    uv run --group notebooks jupyter nbconvert --to notebook --execute --stdout \
        --ExecutePreprocessor.timeout=300 notebooks/*.ipynb > /dev/null

# Clear outputs from every thin notebook in place (run before committing)
notebooks-clear:
    uv run --group notebooks jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
