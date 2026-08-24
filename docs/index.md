---
icon: lucide/brain
---

# Multimodal AD Detection

Code accompanying **"Automated Detection of Alzheimer's Disease: A
Multi-modal Approach With 3D MRI and Amyloid PET"**, published in
*Scientific Reports* (2024).

[:material-file-document: Read the paper](https://www.nature.com/articles/s41598-024-56001-9){ .md-button }
[:material-format-quote-close: How to cite](methodology.md#citation){ .md-button }

## What this is

The paper trains 3D convolutional networks on structural MRI and amyloid
PET volumes from [OASIS-3](https://www.oasis-brains.org/), separately and
via late-fusion, to classify demented vs. non-demented subjects, then uses
Grad-CAM to relate model attention to known AD-affected brain regions.

## Current status: migration in progress

!!! warning "Read this before assuming anything works"
    This repository is being migrated from a set of exploratory Google
    Colab notebooks to a proper `src/` layout Python package
    (`multimodal_ad`). **As of this documentation, no scientific code has
    been ported.** The package is a scaffold: it imports, has a version
    number, and a smoke test. The MRI/PET preprocessing, model
    architectures, and Grad-CAM pipeline described in the paper only exist
    today in the legacy notebooks at the repository root
    (`Dataset_MRI.ipynb`, `Dataset_PET.ipynb`, `Training.ipynb`,
    `Heatmaps.ipynb`, `exploration.ipynb`).

    See [Reproducibility](reproducibility.md) for exactly what does and
    does not run today, and the
    [legacy notebook inventory](legacy-notebooks-inventory.md) for a full
    behavior audit of those notebooks, including open ambiguities that
    must be resolved with the paper authors before porting.

## Where to go next

| I want to... | Go to |
| --- | --- |
| Understand the scientific method | [Methodology](methodology.md) |
| Set up the project locally | [Installation](installation.md) |
| Understand what data I can and can't use | [Data access & contracts](data-access.md) |
| Know what's provenance-checked vs. unknown in committed binary assets | [Asset provenance ledger](asset-provenance.md) |
| Know what's runnable today vs. planned | [Reproducibility](reproducibility.md) |
| Contribute code | [Development & testing](development.md) |
| Browse the package API | [API reference](api.md) |

## Roadmap

1. **Phase 0/1 (done)** — project foundation: `src/` layout, `uv`
   dependency management, lint/type/test tooling, CI, this documentation.
2. **Phase 2 (in progress, separate PR)** — port the data pipeline
   (`Dataset_MRI.ipynb`, `Dataset_PET.ipynb`) into `multimodal_ad`, with
   synthetic fixtures for tests since real OASIS-3 data cannot be
   redistributed (see [Data access & contracts](data-access.md)).
3. **Phase 3 (planned)** — port model architectures and training loop
   (`Training.ipynb`), replacing the Colab/DagsHub-`input()`-based
   experiment tracking with a non-interactive, reproducible setup.
4. **Phase 4 (planned)** — port Grad-CAM interpretability
   (`Heatmaps.ipynb`, `exploration.ipynb`).
5. **Phase 5 (planned)** — replace the legacy `.ipynb` files with thin
   notebooks that call into the ported `multimodal_ad` package, once their
   APIs are stable. This is explicitly **not** done in this documentation
   PR: notebook APIs depend on the data and model ports above.

## Contributing

Read [Development & testing](development.md) for the toolchain
(`uv`, `just`, `ruff`, `pyrefly`, `pytest`, `prek`) and
[AGENTS.md](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/AGENTS.md)
for repository conventions. Before porting any scientific logic, read the
[legacy notebook inventory](legacy-notebooks-inventory.md): it documents
retained constants and open ambiguities that must not be silently changed
or silently guessed at.
