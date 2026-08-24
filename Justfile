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
