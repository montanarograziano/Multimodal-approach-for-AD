# Data access & contracts

## Real data: OASIS-3, external only

This project uses [OASIS-3](https://www.oasis-brains.org/), a
longitudinal neuroimaging dataset distributed under a Data Use Agreement
(DUA) by the OASIS project (Washington University in St. Louis / Knight
ADRC). Key constraints:

- **You must apply for access directly with OASIS.** Neither this
  repository nor its maintainers can grant, proxy, or bundle access to
  OASIS-3 data.
- **Raw or per-subject OASIS-3 data must never be committed to this
  repository**, public or private. This includes clinical CSVs, raw
  `.nii`/`.img`/`.hdr` scans, and any per-subject derived array that could
  be traced back to an individual.
- Aggregate, non-reidentifiable derived artifacts (e.g., population-level
  mean heatmaps across many subjects) are a lower-risk category, but are
  still subject to the DUA's terms on redistribution and must be reviewed
  before committing.
- Code in this repository that expects OASIS-3 data must read it from a
  user-provided local path (never a hardcoded Colab/Drive path, and never
  fetched automatically), consistent with the DUA's terms.

## Synthetic fixtures for tests

Because real OASIS-3 data cannot be committed or fetched in CI, the data
pipeline uses **synthetic fixtures** that match the *shape and dtype
contract* of real data without containing any real scan content:

- `multimodal_ad.data.synthetic.generate_synthetic_dataset` writes
  deterministic (seeded) fake NIfTI volumes — noisy background plus a
  brighter central cube standing in for a brain, so Otsu-threshold brain
  cropping has a real foreground to find — plus a matching manifest CSV.
  Session ids follow the same `<subject>_<modality>_d<days>` shape the
  real OASIS-3 adapter (`multimodal_ad.data.oasis.parse_session_id`)
  expects, so synthetic data exercises the same parsing path real data
  would.
- Every test under `tests/data/` and `tests/models/`, plus the CLI
  quickstart (`python -m multimodal_ad.cli`), runs against this generator
  or small in-memory synthetic arrays/DataFrames, never real scans.

What this does **not** cover: `multimodal_ad.data.oasis`'s CSV column
contract (`OasisLayout`) is a best-effort, explicit schema inferred from
the legacy notebooks, not validated against a real OASIS-3 export (see
that module's docstring and the
[legacy notebook inventory](legacy-notebooks-inventory.md)). Confirming it
against a real export remains open work.

## Non-goals

- This repository will not include a data downloader, scraper, or mirror
  for OASIS-3.
- This repository will not include a de-identification pipeline for
  third-party datasets; users bringing their own data are responsible for
  its compliance.
