# Multimodal AD Detection

Code accompanying **"Automated Detection of Alzheimer's Disease: A
Multi-modal Approach With 3D MRI and Amyloid PET"**, published in
[Scientific Reports (2024)](https://www.nature.com/articles/s41598-024-56001-9)
([doi:10.1038/s41598-024-56001-9](https://doi.org/10.1038/s41598-024-56001-9)).

📖 **[Full documentation site](https://montanarograziano.github.io/Multimodal-approach-for-AD/)**
(also buildable locally, see below).

If you use this work, please cite it, see [`CITATION.cff`](CITATION.cff).

## What this is

The paper trains 3D convolutional networks on structural MRI and amyloid
PET volumes from [OASIS-3](https://www.oasis-brains.org/) to classify
demented vs. non-demented subjects, separately per modality and via
late-fusion, and uses Grad-CAM to relate model attention to known
AD-affected brain regions.

## Current status: data + model pipeline ported, not validated on real data

> [!WARNING]
> This repository is being migrated from a set of exploratory Google
> Colab notebooks to a proper `src/` layout Python package
> (`multimodal_ad`). The typed **data pipeline**
> (`multimodal_ad.data`: manifest, labeling, the OASIS-3 adapter,
> volume preprocessing, augmentation, splitting, and a synthetic
> data generator) and the **model pipeline**
> (`multimodal_ad.models`: 3D CNN builders, training, evaluation,
> Grad-CAM, AAL2 region ranking) have both been ported from the legacy
> notebooks. **What has not happened: validation against real OASIS-3
> data or reproduction of the paper's published metrics.** Every test in
> this repository runs against synthetic, generated-on-the-fly data;
> several implementation ambiguities in the original notebooks (frame
> depth, fusion head shape, CV fold scheme, early-stopping patience,
> diagnosis-labeling edge cases) are resolved here as an explicit,
> documented choice, not a confirmed match to what produced the paper's
> numbers. The five legacy notebooks (plus `images/3D Brain Plot.ipynb`)
> are preserved byte-for-byte as historical artifacts under
> `notebooks/legacy/`; small, newcomer-oriented thin notebooks under
> `notebooks/` now call into the new package on synthetic data (see
> [Notebooks](#notebooks)). See the
> [legacy notebook inventory](docs/legacy-notebooks-inventory.md) for the
> full behavior audit and open ambiguities, and the
> [reproducibility docs](https://montanarograziano.github.io/Multimodal-approach-for-AD/reproducibility/)
> for exactly what does and doesn't run today.

## Repository map

```text
.
├── src/multimodal_ad/
│   ├── data/              # manifest, labeling, OASIS-3 adapter, volumes,
│   │                      # augmentation, splits, synthetic generator
│   ├── models/            # 3D CNN, training, evaluation, Grad-CAM, regions
│   └── cli.py             # synthetic data pipeline quickstart CLI
├── tests/                 # pytest suite (data/, models/, CLI, notebooks, smoke tests)
├── notebooks/
│   ├── 01-data-quickstart.ipynb      # thin: data pipeline on synthetic data
│   ├── 02-tiny-model-workflow.ipynb  # thin: build/train/evaluate/save-reload
│   ├── 03-explainability.ipynb       # thin: Grad-CAM + AAL2 region ranking
│   └── legacy/                       # original Colab notebooks, preserved byte-for-byte
├── docs/                  # documentation site source (Markdown) + notebook inventory
├── zensical.toml          # documentation site config
├── mri1.nii, *.npy        # historical data artifacts, see the legacy notebook inventory
├── samples/               # small figure PNGs referenced by this README
├── AAL2_Atlas_Labels.csv  # AAL2 atlas region labels (region name -> intensity)
├── pyproject.toml, uv.lock
└── Justfile               # `just <recipe>` command shortcuts
```

## Setup

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/) (do not use
`pip`/`poetry`/`pipenv`/`conda` for this project):

```bash
uv sync --locked                 # dev tooling + multimodal_ad.data's runtime deps
uv sync --locked --extra model   # + TensorFlow, for multimodal_ad.models
```

`multimodal_ad.data` and the CLI run without TensorFlow. `multimodal_ad.models`
imports TensorFlow unconditionally, so it's an opt-in extra (`model`), not a
default dependency; see [Model pipeline](#model-pipeline) below.

Common commands (see the [`Justfile`](Justfile)):

```bash
just check           # lint + format check + typecheck + test (CI-equivalent, no TensorFlow needed)
just typecheck-model  # pyrefly check multimodal_ad.models + tests/models (needs `--extra model`)
just test             # pytest
just fmt              # ruff format
just lint             # ruff check
just hooks            # run all prek hooks
just docs-serve       # live-reload documentation preview
just docs-build       # strict documentation build
just notebooks-launch  # launch Jupyter against notebooks/
just notebooks-execute # execute every thin notebook from a clean kernel
```

Full command reference:
[installation docs](https://montanarograziano.github.io/Multimodal-approach-for-AD/installation/).

## Quickstart: synthetic data pipeline

No OASIS-3 access needed. Generates a small deterministic synthetic
dataset (fake NIfTI volumes with a brighter central "brain" blob) and runs
it through manifest construction, subject-wise splitting, and volume
preprocessing, CPU-only and fast enough for a laptop:

```bash
uv run python -m multimodal_ad.cli --output-dir /tmp/synthetic-data --n-subjects 6
```

This is a pipeline smoke test, not a training run: it exercises the same
code paths real OASIS-3 data would go through, on data with no clinical
meaning. See [`src/multimodal_ad/cli.py`](src/multimodal_ad/cli.py) and
`python -m multimodal_ad.cli --help` for options.

## Notebooks

`notebooks/*.ipynb` are small, newcomer-oriented notebooks that call into
`multimodal_ad` on deterministic synthetic data, no OASIS-3 access, no
Colab, no manual paths:

- [`01-data-quickstart.ipynb`](notebooks/01-data-quickstart.ipynb): generate
  a synthetic manifest/dataset, validate/split/process it, visualize a
  central slice.
- [`02-tiny-model-workflow.ipynb`](notebooks/02-tiny-model-workflow.ipynb):
  build a reduced-filter 3D model, one bounded train/evaluate/save-reload
  flow, correct metrics (needs the `model` extra).
- [`03-explainability.ipynb`](notebooks/03-explainability.ipynb): native
  Grad-CAM and AAL2 region ranking on synthetic/toy arrays (needs the
  `model` extra).

```bash
uv sync --locked --extra model --group notebooks
just notebooks-launch
```

All reusable logic lives in `src/`; these notebooks only orchestrate and
display. Commit with cleared outputs (`just notebooks-clear`); CI
re-executes every notebook from a clean kernel. The original Colab
notebooks are preserved byte-for-byte under `notebooks/legacy/` (see the
[legacy notebook inventory](docs/legacy-notebooks-inventory.md)); they are
not executable outside their original Colab/Drive environment.

## Data access

Real data comes from **OASIS-3**, distributed under a Data Use Agreement.
Apply for access directly at [oasis-brains.org](https://www.oasis-brains.org/);
neither this repository nor its maintainers can grant or proxy access.
Real per-subject OASIS-3 data must never be committed here.
`multimodal_ad.data.oasis` is a thin adapter over a **locally provided**
OASIS-3 export (it validates an expected file layout and reads CSVs/scans
the user already has on disk; it never downloads or mirrors anything).
Every test in this repository, including for the OASIS adapter, uses
synthetic fixtures matching the real data's shape/dtype contract
(`multimodal_ad.data.synthetic`), never real scans. See
[data access & contracts](https://montanarograziano.github.io/Multimodal-approach-for-AD/data-access/).

## Model pipeline

`multimodal_ad.models` ports the 3D CNN architectures, training loop,
evaluation metrics, Grad-CAM, and AAL2 region ranking from `Training.ipynb`
and `Heatmaps.ipynb`/`exploration.ipynb`. It requires the `model` extra:

```bash
uv sync --locked --extra model
just typecheck-model
uv run pytest tests/models
```

Highlights (see [API reference](https://montanarograziano.github.io/Multimodal-approach-for-AD/api/)
for full signatures):

- `models.architecture.build_3d_cnn` / `Cnn3DConfig`: the paper's four
  `Conv3D → MaxPool3D → BatchNorm` blocks + dense head, parameterized
  instead of copy-pasted per notebook variant; `build_fusion_model` for
  the late-fusion dual-input model.
- `models.training.train_model` / `TrainingConfig`: compiles, seeds
  (Python/NumPy/TensorFlow), trains with the notebook's retained
  optimizer/schedule/early-stopping constants, and restores the best
  checkpoint. No MLflow/DagsHub logging (dropped, see module docstring).
- `models.evaluation.evaluate_predictions`: accuracy/sensitivity/
  specificity/AUC from thresholded probabilities, with explicit
  zero-class edge-case handling (`nan`, not a crash).
- `models.gradcam.make_gradcam_heatmap`: a native `tf.GradientTape`-based
  3D Grad-CAM (no `tf-keras-vis` dependency).
- `models.regions.rank_regions`: AAL2 atlas region-importance ranking from
  a Grad-CAM heatmap (requires a separately sourced `atlas.nii.gz`, not
  checked into this repo).

## Fast checks vs. model checks

- `just check` (lint, format check, `pyrefly check`, `pytest`) runs
  against `multimodal_ad.data` and the CLI only, with no TensorFlow
  needed; this is the default/fast CI job.
- `just typecheck-model` and the `tests/models/` suite need
  `uv sync --extra model` (TensorFlow) first; CI runs these in a separate
  `model-smoke` job.
- `pyrefly`'s default scope (`pyproject.toml`, `[tool.pyrefly]`)
  explicitly excludes `src/multimodal_ad/models` and `tests/models` so the
  fast job doesn't require TensorFlow to type-check.

## Reproducibility limits

`just check`, the model-extra typecheck/tests, and the CLI quickstart all
run cleanly on a clean clone, CPU-only, no OASIS-3 data required. That is
the extent of what's verified. **This repository does not reproduce the
paper's published metrics and has not been validated against real
OASIS-3 data.** Several implementation ambiguities across notebook cells
(frame depth, fusion head shape, CV fold scheme, early-stopping patience,
diagnosis-labeling edge cases) are resolved here as documented,
best-effort choices, not confirmed matches to what produced the paper's
numbers; see each module's docstring (`multimodal_ad.data.volumes`,
`multimodal_ad.models.architecture`, etc.) for the specific choice made
and why. Full details:
[reproducibility docs](https://montanarograziano.github.io/Multimodal-approach-for-AD/reproducibility/).

## Roadmap

1. **Phase 0/1 (done)**: `src/` layout, `uv`, lint/type/test tooling, CI,
   documentation.
2. **Phase 2a (done)**: typed data pipeline (`multimodal_ad.data`) ported
   from `Dataset_MRI.ipynb`/`Dataset_PET.ipynb`, with a synthetic
   NIfTI/manifest generator for tests.
3. **Phase 2b (done)**: model training/evaluation/Grad-CAM/region-ranking
   (`multimodal_ad.models`) ported from `Training.ipynb`/
   `Heatmaps.ipynb`/`exploration.ipynb`.
4. **Phase 3 (done)**: added thin notebooks under `notebooks/` calling
   into the stable `multimodal_ad` API on synthetic data, and moved the
   original Colab notebooks to `notebooks/legacy/` (preserved
   byte-for-byte). Real-OASIS-3 validation/metric reproduction, with the
   paper authors' input on the open ambiguities tracked in the
   [legacy notebook inventory](docs/legacy-notebooks-inventory.md), remains
   future work.

## Contributing

Read [`AGENTS.md`](AGENTS.md) for repository conventions and the
[development docs](https://montanarograziano.github.io/Multimodal-approach-for-AD/development/)
for the full toolchain. Before touching any scientific logic, read the
[legacy notebook inventory](docs/legacy-notebooks-inventory.md): it
documents retained constants and open ambiguities that must not be
silently changed or guessed at.

## License

[MIT](LICENSE) for the code in this repository. This does **not** cover
OASIS-3 data, which remains subject to its own Data Use Agreement.
