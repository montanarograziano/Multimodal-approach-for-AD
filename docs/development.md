# Development & testing

## Toolchain

- Package manager: `uv` (`pyproject.toml` + `uv.lock`).
- Formatter/linter: `ruff` (`ruff format`, `ruff check`), configured in
  `pyproject.toml` (`line-length = 100`, `target-version = "py313"`,
  rule sets `E, F, I, UP, B, SIM, N`).
- Type checker: `pyrefly`, scoped to `src/` and `tests/`.
- Test runner: `pytest`, with coverage (`--cov=multimodal_ad`).
- Git hooks: `prek` running the hooks in `.pre-commit-config.yaml`.

See [Installation](installation.md) for setup and the command table.

## Repository layout

```text
.
├── src/multimodal_ad/     # importable package (scaffold today)
├── tests/                 # pytest suite
├── docs/                  # this documentation site's Markdown source
├── zensical.toml          # documentation site config
├── *.ipynb                # legacy Colab notebooks (excluded from lint/format)
├── mri1.nii, *.npy        # historical data artifacts, see asset provenance ledger
├── samples/                # small figure PNGs
├── images/                # separate Poetry sub-project for paper figures
├── AAL2_Atlas_Labels.csv  # AAL2 atlas region labels
├── pyproject.toml, uv.lock
└── Justfile
```

## Conventions

- Add heavy scientific dependencies (TensorFlow/Keras, NumPy, OpenCV,
  nibabel, ...) to `[project.optional-dependencies].science`, not
  `[project.dependencies]`, until `multimodal_ad` actually imports them.
- Add dev tooling to `[dependency-groups].dev`, notebook-only tooling to
  `[dependency-groups].notebooks`, and documentation tooling to
  `[dependency-groups].docs`.
- `.ipynb` files are excluded from Ruff lint/format
  (`extend-exclude = ["*.ipynb"]`) and are preserved as historical
  artifacts. Don't edit notebook cells without an explicit task to do so.
- Type hints are required on new code in `src/`.
- Every non-trivial change needs a test in `tests/`. Prefer plain
  `assert`-based tests; avoid fixtures/frameworks beyond pytest unless the
  code under test needs them.

## Before committing

```bash
just check   # lint + format check + typecheck + test
just hooks   # run prek hooks
```

Full details and rationale live in
[AGENTS.md](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/AGENTS.md),
which takes precedence over this page if the two ever disagree.

## Documentation site

The docs you're reading are built with [Zensical](https://zensical.org/)
from Markdown files in `docs/`, configured in `zensical.toml` at the
repository root.

```bash
just docs-serve   # live-reload preview at http://localhost:8000
just docs-build   # strict build (warnings fail the build) into ./site
```

`site/` is git-ignored; it is a build artifact, not committed source.

## Notebooks are not yet ported to "thin" wrappers

A common target-state pattern is for `.ipynb` files to become thin,
mostly-markdown notebooks that import and call a stable `multimodal_ad`
API. That is **explicitly deferred**: the data-loading and model APIs the
notebooks would call don't exist yet, and are being designed in the
data-pipeline and training-port PRs. Rewriting notebooks against an API
that doesn't exist yet would produce speculative, likely-wrong code.
