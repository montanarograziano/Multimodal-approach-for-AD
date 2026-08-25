# API reference

This page is hand-maintained, manually linked to source, not
auto-generated. `mkdocstrings`/Zensical's API-doc integration was
evaluated and not wired up here: this project's public surface is still
small enough that accurate manual references are cheaper and less risky
than adding and verifying a new doc-generation dependency (see
[AGENTS.md](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/AGENTS.md)'s
KISS/YAGNI stance on new tooling). Revisit if the package surface grows
enough that this page visibly drifts from the code.

For the definitive signatures/behavior, read the linked source: every
public function and class below has a docstring, and most modules have an
extensive module-level docstring documenting divergences from the legacy
notebooks. `import multimodal_ad; multimodal_ad.__version__` also still
works (currently `0.1.0`).

## `multimodal_ad.data` (no TensorFlow required)

[`src/multimodal_ad/data/`](https://github.com/montanarograziano/Multimodal-approach-for-AD/tree/main/src/multimodal_ad/data)

| Module | Key API | Purpose |
| --- | --- | --- |
| [`manifest`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/data/manifest.py) | `Modality`, `ScanRecord`, `ScanManifest` | Canonical typed scan manifest / data contract (CSV round-trip via `to_dataframe`/`from_dataframe`, `to_csv`). |
| [`labeling`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/data/labeling.py) | `classify_diagnosis`, `classify_diagnosis_row`, `smooth_temporal_labels`, `label_nearest_visit` | Diagnosis normalization and longitudinal (temporal) label smoothing, nearest-clinical-visit labeling. |
| [`oasis`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/data/oasis.py) | `OasisLayout`, `validate_layout`, `parse_session_id`, `load_clinical_table`, `load_scan_index` | Adapter for a **locally provided** OASIS-3 export layout; never downloads or redistributes data. |
| [`volumes`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/data/volumes.py) | `ProcessingConfig`, `process_scan`, `process_volume`, `load_volume`, `find_brain_bounding_box`, `normalize_intensity` | MRI/PET loading, 4D frame averaging, normalize → central-slice → Otsu brain-crop → resize, in that fidelity-checked order. |
| [`augmentation`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/data/augmentation.py) | `augment_volume` | Deterministic (caller-seeded `numpy.random.Generator`) rotate/flip augmentation on raw, pre-`process_volume` volumes. |
| [`splits`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/data/splits.py) | `subject_train_test_split`, `stratified_subject_folds`, `assert_no_subject_leakage` | Subject-wise (not scan-wise) splitting/folding with an explicit leakage check. |
| [`synthetic`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/data/synthetic.py) | `SyntheticDatasetConfig`, `generate_synthetic_dataset`, `load_synthetic_manifest` | Deterministic synthetic NIfTI + manifest generator; what every test and the CLI quickstart run against. |

```python
from pathlib import Path
from multimodal_ad.data.synthetic import generate_synthetic_dataset
from multimodal_ad.data.splits import subject_train_test_split

manifest = generate_synthetic_dataset(Path("/tmp/synthetic-data"))
train_df, test_df = subject_train_test_split(manifest.to_dataframe(), seed=1234)
```

## `multimodal_ad.models` (requires the `model` extra, e.g. `just install`)

[`src/multimodal_ad/models/`](https://github.com/montanarograziano/Multimodal-approach-for-AD/tree/main/src/multimodal_ad/models)
imports TensorFlow/Keras unconditionally; every symbol below is
unavailable until the `model` extra is installed.

| Module | Key API | Purpose |
| --- | --- | --- |
| [`architecture`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/models/architecture.py) | `Cnn3DConfig`, `build_3d_cnn`, `build_mri_input_model`, `build_pet_input_model`, `build_fusion_model`, `extract_feature_extractor` | The paper's 3D CNN topology, parameterized over the notebook's four near-duplicate model-builder functions; late-fusion dual-input model. |
| [`training`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/models/training.py) | `TrainingConfig`, `TrainingResult`, `train_model`, `seed_everything`, `save_model`, `load_model` | Compile/fit/evaluate with the notebook's retained optimizer/schedule/early-stopping constants; deterministic seeding; no MLflow. |
| [`evaluation`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/models/evaluation.py) | `EvaluationMetrics`, `evaluate_predictions`, `threshold_probabilities`, `sensitivity`, `specificity` | Accuracy/sensitivity/specificity/AUC from thresholded probabilities, with explicit `nan` handling for zero-class edge cases. |
| [`gradcam`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/models/gradcam.py) | `make_gradcam_heatmap`, `last_conv_layer_name`, `unwrap_output_activation` | Native `tf.GradientTape`-based 3D Grad-CAM (no `tf-keras-vis` dependency). |
| [`regions`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/models/regions.py) | `load_atlas`, `pad_to_frame`, `load_region_labels`, `rank_regions` | AAL2 atlas region-importance ranking from a Grad-CAM heatmap; the atlas (`atlas.nii.gz`) and its region labels are bundled at the repo root. |

```python
from multimodal_ad.models.architecture import Cnn3DConfig, build_3d_cnn
from multimodal_ad.models.training import TrainingConfig, train_model

model = build_3d_cnn(Cnn3DConfig(width=32, height=32, depth=16))  # small, CPU-fast shape
result = train_model(model, x_train, y_train, x_val, y_val, TrainingConfig(epochs=2))
print(result.val_accuracy)
```

## `multimodal_ad.cli`

[`src/multimodal_ad/cli.py`](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/src/multimodal_ad/cli.py):
`build_parser`, `run`, `main`. A thin `argparse` wrapper (not a framework)
around `data.synthetic`/`data.splits`/`data.volumes` for the quickstart
described in the [README](https://github.com/montanarograziano/Multimodal-approach-for-AD/blob/main/README.md#quickstart-synthetic-data-pipeline).
Run `python -m multimodal_ad.cli --help` for options.

## What's *not* covered here

The legacy notebooks still implement some things ad hoc that haven't been
promoted to a typed API (e.g., a driver that wires `data`+`models`+`regions`
into one end-to-end training/evaluation/interpretability run across CV
folds). See the [legacy notebook inventory](legacy-notebooks-inventory.md)
for what those cells do today, and [Reproducibility](reproducibility.md)
for what's verified to run versus not.
