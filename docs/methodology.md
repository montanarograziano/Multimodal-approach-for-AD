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

## Ported, with documented divergences and open ambiguities

`multimodal_ad.data` and `multimodal_ad.models` now implement the above
(see [API reference](api.md)), but several implementation details the
legacy notebooks left ambiguous across cells (frame depth, fusion head
shape, early-stopping patience, CV fold scheme, diagnosis-regex edge
cases) are resolved as an explicit, documented default per module rather
than a confirmed match to whatever produced the paper's published numbers.
For example:

- **Frame depth**: `data.volumes.ProcessingConfig` defaults to
  `n_frames=50` (matching the paper's reported `(128, 128, 50)` input),
  exposed as a config field since the notebooks also use 20 and 30 in
  other cells.
- **Fusion head**: `models.architecture.build_fusion_model` implements the
  non-buggy of `Training.ipynb`'s two conflicting `get_merged()`
  definitions (`Dense(128, relu)` head); the other definition discards an
  unused `Dense(4, relu)` layer via an apparent copy-paste bug.
- **Grad-CAM**: `models.gradcam` ports the notebook's unused, hand-written
  `tf.GradientTape`-based recipe (generalized to 3D) instead of the
  actually-used `tf-keras-vis` path, which relied on a fragile numeric
  layer offset (`penultimate_layer=-7`); `tf-keras-vis` is no longer a
  dependency.
- **Experiment tracking**: the notebooks' interactive, `input()`-prompted
  MLflow/DagsHub tracking is dropped entirely, not replaced; `train_model`
  returns typed results (history, evaluation metrics) instead.

See each module's docstring
(`multimodal_ad.data.volumes`, `.augmentation`, `.oasis`;
`multimodal_ad.models.architecture`, `.training`, `.gradcam`, `.regions`)
for the full reasoning behind each choice, and the
["Summary: what must be resolved before Phase 2"](legacy-notebooks-inventory.md#summary-what-must-be-resolved-before-phase-2-scientific-code-porting)
section of the notebook inventory for the ambiguities that still need the
paper authors' input before any of this can be called a confirmed
reproduction.
