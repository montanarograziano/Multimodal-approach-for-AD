# API reference

## Current state: scaffold only

`multimodal_ad` currently exposes no public API beyond a package
docstring and `__version__`:

```python
>>> import multimodal_ad
>>> multimodal_ad.__version__
'0.1.0'
```

There is nothing to document yet: no dataset loaders, no model
definitions, no training or evaluation entry points. Those are being
designed and ported in separate, stacked PRs (see
[the roadmap](index.md#roadmap)).

## Plan for this page

Once `multimodal_ad` gains real modules (data pipeline, models,
interpretability), this page is expected to switch to an auto-generated
reference (for example via `mkdocstrings` or an equivalent tool that reads
docstrings and type hints directly from `src/multimodal_ad`), rather than
hand-maintained prose that will drift from the code. That tooling is not
wired up in this PR because there is no source to generate from yet, and
adding it now would be undocumented, untestable scaffolding.

## In the meantime

- For the *scientific* API the notebooks currently implement ad hoc (data
  loading, preprocessing, model construction), see the
  [legacy notebook inventory](legacy-notebooks-inventory.md), which
  documents function names, signatures, and behavior as they exist today.
- For what's actually importable and tested right now, see
  [`tests/test_package.py`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/tests/test_package.py).
