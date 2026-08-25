# Development & testing

## Toolchain

- Package manager: `uv` (`pyproject.toml` + `uv.lock`).
- Formatter/linter: `ruff` (`ruff format`, `ruff check`), configured in
  `pyproject.toml` (`line-length = 100`, `target-version = "py313"`,
  rule sets `E, F, I, UP, B, SIM, N`).
- Type checker: `pyrefly`, scoped to `src/`/`tests/` by default,
  excluding `multimodal_ad.models`/`tests/models` (those import
  TensorFlow unconditionally); `just typecheck-model` checks that subtree
  separately once the `model` extra is installed (`just install`).
- Test runner: `pytest`, with coverage (`--cov=multimodal_ad`). Tests
  under `tests/models/` use `pytest.importorskip("tensorflow")` and
  auto-skip if the `model` extra isn't installed.
- Git hooks: `prek` running the hooks in `.pre-commit-config.yaml`.

See [Installation](installation.md) for setup and the command table.

## Repository layout

```text
.
├── src/multimodal_ad/
│   ├── data/              # manifest, labeling, OASIS-3 adapter, volumes,
│   │                      # augmentation, splits, synthetic generator
│   ├── models/            # 3D CNN, training, evaluation, Grad-CAM, regions
│   └── cli.py             # synthetic data pipeline quickstart CLI
├── tests/                 # pytest suite (data/, models/, CLI, notebooks, smoke tests)
├── notebooks/             # thin notebooks calling into multimodal_ad (synthetic data)
│   └── legacy/            # original Colab notebooks, preserved byte-for-byte
├── docs/                  # this documentation site's Markdown source
├── zensical.toml          # documentation site config
├── mri1.nii, *.npy        # historical data artifacts, see the legacy notebook inventory
├── samples/                # small figure PNGs
├── atlas.nii.gz           # AAL2 atlas volume, bundled
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
- All `.ipynb` files are excluded from Ruff lint/format
  (`extend-exclude = ["*.ipynb"]`). `notebooks/legacy/*.ipynb` are
  preserved as historical artifacts; don't edit their cells without an
  explicit task to do so. `notebooks/*.ipynb` must stay thin (orchestration
  only, no function/class definitions, no Colab/shell/credential/absolute-
  path patterns, cleared outputs); see `tests/notebooks/` and "Notebooks"
  below.
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

## Notebooks

`notebooks/*.ipynb` are thin, newcomer-oriented notebooks that import and
call the stable `multimodal_ad` API on deterministic synthetic data (data
pipeline, tiny CPU model workflow, explainability); all reusable logic
lives in `src/`. `notebooks/legacy/*.ipynb` are the original five Colab
notebooks plus `images/3D Brain Plot.ipynb`, preserved byte-for-byte (see
the [legacy notebook inventory](legacy-notebooks-inventory.md)) and not
executable outside their original Colab/Drive environment.

```bash
just install
just notebooks-launch    # launch Jupyter against notebooks/
just notebooks-execute   # execute every thin notebook from a clean kernel
just notebooks-clear     # clear outputs before committing
```

`tests/notebooks/` enforces thinness (no function/class definitions),
forbidden patterns (Colab mounts, shell installs, absolute paths,
credentials), and cleared outputs on every commit; the `notebook-smoke` CI
job additionally re-executes every notebook from a clean kernel, CPU-only.
Real-OASIS-3 validation of the underlying `multimodal_ad` pipeline (not a
notebook concern) remains follow-up work; see the
[README roadmap](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/README.md#roadmap).
