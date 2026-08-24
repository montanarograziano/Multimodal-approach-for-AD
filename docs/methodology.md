# Methodology

This page summarizes the published method. It describes **what the paper
reports**, not necessarily what today's code in this repository does; see
[Reproducibility](reproducibility.md) and the
[legacy notebook inventory](legacy-notebooks-inventory.md) for the gap
between the two.

## Paper

Castellano, G., Esposito, A., Lella, E., Montanaro, G., & Vessio, G. (2024).
*Automated detection of Alzheimer's disease: a multi-modal approach with 3D
MRI and amyloid PET.* Scientific Reports, 14, 5210.
[doi:10.1038/s41598-024-56001-9](https://doi.org/10.1038/s41598-024-56001-9)

## Data

- **Source**: [OASIS-3](https://www.oasis-brains.org/), a longitudinal
  neuroimaging dataset of structural MRI and amyloid PET (PIB/AV45 tracers)
  scans with associated clinical diagnoses.
- **Labeling**: subjects are labeled demented / non-demented from
  differential-diagnosis fields nearest in time to each scan, with a
  temporal smoothing step to reduce visit-level noise.
- **Balancing**: negative class undersampled, positive class augmented via
  rotation/flip to counter OASIS-3's class imbalance (dementia is a
  minority outcome in a general longitudinal cohort).
- **Preprocessing**: brain bounding-box extraction (Otsu thresholding),
  central-frame extraction, resize to a common shape, min-max intensity
  normalization.

## Model

- A 3D CNN (four `Conv3D → MaxPool3D → BatchNorm` blocks, `GlobalAveragePooling3D`,
  dense head), trained separately on MRI and PET.
- Late-fusion transfer learning between modalities, and a merged dual-input
  model concatenating penultimate-layer features from both single-modality
  models.
- Repeated k-fold cross-validation with confusion-matrix-derived metrics
  (accuracy, sensitivity, specificity, F1).

## Interpretability

- Grad-CAM saliency maps (via
  [`tf-keras-vis`](https://github.com/keisen/tf-keras-vis)) computed
  per-class, averaged across samples into population-level heatmaps.
- Heatmap importance mapped onto AAL2 atlas regions to rank anatomical
  zones by mean attention, relating model behavior to known
  AD-affected regions.

## Citation

If you use this work, please cite the paper (preferred) or the software,
per [`CITATION.cff`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/CITATION.cff):

```bibtex
@article{castellano2024automated,
  title   = {Automated detection of {A}lzheimer's disease: a multi-modal approach with {3D} {MRI} and amyloid {PET}},
  author  = {Castellano, Giovanna and Esposito, Andrea and Lella, Eufemia and Montanaro, Graziano and Vessio, Gennaro},
  journal = {Scientific Reports},
  volume  = {14},
  pages   = {5210},
  year    = {2024},
  doi     = {10.1038/s41598-024-56001-9}
}
```

## What is *not* yet ported

The exact retained implementation details (frame depth, fusion head shape,
early-stopping patience, CV fold scheme, diagnosis-regex edge cases) have
open ambiguities across notebook cells that must be resolved with the
paper authors before Phase 2+ ports them into `multimodal_ad`. See the
["Summary: what must be resolved before Phase 2"](legacy-notebooks-inventory.md#summary-what-must-be-resolved-before-phase-2-scientific-code-porting)
section of the notebook inventory.
