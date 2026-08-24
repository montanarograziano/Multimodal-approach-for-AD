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

## Current status: data + model pipeline ported, not validated on real data

!!! warning "Read this before assuming results are reproduced"
    This repository is being migrated from a set of exploratory Google
    Colab notebooks to a proper `src/` layout Python package
    (`multimodal_ad`). The **data pipeline** (`multimodal_ad.data`:
    manifest, labeling, the local OASIS-3 adapter, volume preprocessing,
    augmentation, subject-wise splitting, and a synthetic data generator)
    and the **model pipeline** (`multimodal_ad.models`: 3D CNN builders,
    training, evaluation, Grad-CAM, AAL2 region ranking) have both been
    ported from the legacy notebooks.

    **What has not happened**: validation against real OASIS-3 data, or
    reproduction of the paper's published metrics. Every automated test
    here runs against synthetic, generated-on-the-fly data (see
    [Data access & contracts](data-access.md)). Several implementation
    ambiguities in the original notebooks (frame depth, fusion head
    shape, CV fold scheme, early-stopping patience, diagnosis-labeling
    edge cases) are resolved as an explicit, documented choice per
    module, not a confirmed match to whatever produced the paper's
    numbers.

    See [Reproducibility](reproducibility.md) for exactly what does and
    does not run today, and the
    [legacy notebook inventory](legacy-notebooks-inventory.md) for the
    full behavior audit and open ambiguities that still need the paper
    authors' input.

## Where to go next

| I want to... | Go to |
| --- | --- |
| Understand the scientific method | [Methodology](methodology.md) |
| Set up the project locally | [Installation](installation.md) |
| Understand what data I can and can't use | [Data access & contracts](data-access.md) |
| Know what's runnable today vs. planned | [Reproducibility](reproducibility.md) |
| Contribute code | [Development & testing](development.md) |
| Browse the package API | [API reference](api.md) |

## Roadmap

1. **Phase 0/1 (done)** — project foundation: `src/` layout, `uv`
   dependency management, lint/type/test tooling, CI, this documentation.
2. **Phase 2a (done)** — typed data pipeline (`multimodal_ad.data`)
   ported from `Dataset_MRI.ipynb`/`Dataset_PET.ipynb`, with a synthetic
   NIfTI/manifest generator so tests never need real OASIS-3 data (see
   [Data access & contracts](data-access.md)).
3. **Phase 2b (done)** — model architectures, training loop, evaluation
   metrics, Grad-CAM, and AAL2 region ranking (`multimodal_ad.models`)
   ported from `Training.ipynb`/`Heatmaps.ipynb`/`exploration.ipynb`,
   with MLflow/DagsHub's interactive tracking dropped rather than
   replaced (see [Methodology](methodology.md)).
4. **Phase 3 (done)** — thin notebooks under `notebooks/` that call into
   the `multimodal_ad` API on synthetic data; the five legacy notebooks
   (plus `images/3D Brain Plot.ipynb`) moved byte-for-byte to
   `notebooks/legacy/`. Real-OASIS-3 validation/metric reproduction
   remains future work, pending resolution of the open ambiguities in the
   [legacy notebook inventory](legacy-notebooks-inventory.md#summary-what-must-be-resolved-before-phase-2-scientific-code-porting)
   with the paper authors.

## Contributing

Read [Development & testing](development.md) for the toolchain
(`uv`, `just`, `ruff`, `pyrefly`, `pytest`, `prek`) and
[AGENTS.md](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/AGENTS.md)
for repository conventions. Before porting any scientific logic, read the
[legacy notebook inventory](legacy-notebooks-inventory.md): it documents
retained constants and open ambiguities that must not be silently changed
or silently guessed at.
