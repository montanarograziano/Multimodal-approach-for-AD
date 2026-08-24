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

## Current status: migration in progress

> [!WARNING]
> This repository is being migrated from a set of exploratory Google
> Colab notebooks to a proper `src/` layout Python package
> (`multimodal_ad`). **As of this README, no scientific code has been
> ported.** The package is a scaffold (it imports, has a version number,
> and a smoke test). The MRI/PET preprocessing, model architectures, and
> Grad-CAM pipeline described in the paper exist today only in the legacy
> notebooks at the repository root: `Dataset_MRI.ipynb`,
> `Dataset_PET.ipynb`, `Training.ipynb`, `Heatmaps.ipynb`,
> `exploration.ipynb`. None of them run end-to-end outside the original
> author's interactive Colab session (hardcoded placeholder paths,
> Colab-only imports, undefined names referenced across cells). See the
> [legacy notebook inventory](docs/legacy-notebooks-inventory.md) for a
> full behavior audit, and the
> [reproducibility docs](https://montanarograziano.github.io/Multimodal-approach-for-AD/reproducibility/)
> for exactly what does and doesn't run today.

## Repository map

```text
.
├── src/multimodal_ad/     # importable package (scaffold today)
├── tests/                 # pytest suite
├── docs/                  # documentation site source (Markdown) + notebook inventory
├── zensical.toml          # documentation site config
├── Dataset_MRI.ipynb      # legacy: MRI dataset construction (not yet ported)
├── Dataset_PET.ipynb      # legacy: PET dataset construction (not yet ported)
├── Training.ipynb         # legacy: model training (not yet ported)
├── Heatmaps.ipynb         # legacy: Grad-CAM heatmaps (not yet ported)
├── exploration.ipynb      # legacy: AAL2 zone ranking (not yet ported)
├── images/                # separate Poetry sub-project generating paper figures
├── mri1.nii, *.npy        # historical data artifacts — see the asset provenance ledger
├── samples/               # small figure PNGs referenced by this README
├── AAL2_Atlas_Labels.csv  # AAL2 atlas region labels
├── pyproject.toml, uv.lock
└── Justfile               # `just <recipe>` command shortcuts
```

## Setup

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/) (do not use
`pip`/`poetry`/`pipenv`/`conda` for this project):

```bash
uv sync --locked                       # dev tooling only
uv sync --locked --extra science       # + TensorFlow/OpenCV/nibabel/etc.
```

Common commands (see the [`Justfile`](Justfile)):

```bash
just check        # lint + format check + typecheck + test (CI-equivalent)
just test          # pytest
just fmt           # ruff format
just lint          # ruff check
just typecheck     # pyrefly check
just hooks         # run all prek hooks
just docs-serve    # live-reload documentation preview
just docs-build    # strict documentation build
```

Full command reference:
[installation docs](https://montanarograziano.github.io/Multimodal-approach-for-AD/installation/).

## Data access

Real data comes from **OASIS-3**, distributed under a Data Use Agreement.
Apply for access directly at [oasis-brains.org](https://www.oasis-brains.org/);
neither this repository nor its maintainers can grant or proxy access.
Real per-subject OASIS-3 data must never be committed here. Tests for the
ported data pipeline are expected to use synthetic fixtures matching the
real data's shape/dtype contract, not real scans. See
[data access & contracts](https://montanarograziano.github.io/Multimodal-approach-for-AD/data-access/)
and the
[asset provenance ledger](https://montanarograziano.github.io/Multimodal-approach-for-AD/asset-provenance/)
(which flags unresolved-provenance binary artifacts already in this
repository's history).

## Reproducibility limits

Today: lint/format/typecheck/test pass on a clean clone with no data.
Nothing else. The paper's pipeline is not runnable end-to-end from this
repository yet, and several implementation ambiguities (frame depth,
fusion head shape, CV fold scheme, early-stopping patience, diagnosis
labeling edge cases) exist across notebook cells and are not yet resolved
with the paper authors. Full details:
[reproducibility docs](https://montanarograziano.github.io/Multimodal-approach-for-AD/reproducibility/).

## Roadmap

1. **Phase 0/1 (done)**: `src/` layout, `uv`, lint/type/test tooling, CI,
   documentation (this PR).
2. **Phase 2 (separate, parallel PR)**: port the data pipeline
   (`Dataset_MRI.ipynb`, `Dataset_PET.ipynb`) with synthetic test
   fixtures.
3. **Phase 3 (planned)**: port model training (`Training.ipynb`) with
   reproducible, non-interactive experiment tracking.
4. **Phase 4 (planned)**: port Grad-CAM interpretability
   (`Heatmaps.ipynb`, `exploration.ipynb`).
5. **Phase 5 (planned)**: replace legacy `.ipynb` files with thin
   notebooks calling into the ported package, once its API is stable.

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
