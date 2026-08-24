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
  before committing. See the
  [asset provenance ledger](asset-provenance.md) for the artifacts already
  in this repository's history, several of which predate this policy and
  have **unresolved provenance**.
- Code in this repository that expects OASIS-3 data must read it from a
  user-provided local path (never a hardcoded Colab/Drive path, and never
  fetched automatically), consistent with the DUA's terms.

## Synthetic fixtures for tests

Because real OASIS-3 data cannot be committed or fetched in CI, tests for
the data pipeline (Phase 2+) are expected to use **synthetic fixtures**
that match the *shape and dtype contract* of real data without containing
any real scan content, for example:

- Randomly generated 3D volumes with the same shape as pipeline output
  (`(128, 128, 50)`, per the legacy notebooks) and matching dtype
  (`float32` after normalization).
- Synthetic clinical CSVs with the same columns the labeling code expects
  (`Subject`, `Date`, `dx1`..`dx5`, ...), populated with fabricated
  subject IDs and dates, sized to exercise edge cases (e.g., a subject
  with only one visit, a subject whose diagnosis flips between visits).

**This repository does not yet contain such fixtures.** They are the
responsibility of the data-pipeline port (a separate, parallel PR) once
the pipeline code they exercise exists. This page documents the contract
that PR is expected to satisfy, not a shipped feature.

## Non-goals

- This repository will not include a data downloader, scraper, or mirror
  for OASIS-3.
- This repository will not include a de-identification pipeline for
  third-party datasets; users bringing their own data are responsible for
  its compliance.
