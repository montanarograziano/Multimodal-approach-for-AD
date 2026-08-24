"""Typed data pipeline for MRI/PET scan ingestion, labeling, and splitting.

Ports the scientifically meaningful behavior of `Dataset_MRI.ipynb` /
`Dataset_PET.ipynb` (see `docs/legacy-notebooks-inventory.md`) into current,
typed Python. Modules:

- `manifest`: the canonical typed scan manifest / data contract.
- `labeling`: diagnosis normalization and longitudinal (temporal) label
  correction, nearest-clinical-visit labeling.
- `oasis`: adapter for a locally provided OASIS-3 directory layout (no
  download/redistribution).
- `volumes`: MRI/PET volume loading, 4D frame averaging, central axial
  slicing, brain bounding/crop/resize, intensity normalization.
- `augmentation`: deterministic, seeded volume augmentation.
- `splits`: subject-wise train/test and stratified fold creation with
  leakage checks.
- `synthetic`: deterministic synthetic NIfTI + manifest generator so the
  full data path is exercisable without real OASIS-3 data.
"""
