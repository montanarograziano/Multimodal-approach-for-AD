"""Model/training/evaluation/explainability pipeline (Phase 2b).

Ports the scientifically meaningful behavior of `Training.ipynb` and
`Heatmaps.ipynb` (see `docs/legacy-notebooks-inventory.md`) into current,
typed Python. **Imports TensorFlow unconditionally**; install the `model`
extra (`uv sync --extra model`) to use this subpackage. `multimodal_ad.data`
and the CLI do not depend on it. Modules:

- `architecture`: the 3D CNN builder (`build_3d_cnn`) and its frozen/named
  fusion-branch/fusion-head variants, parameterizing the four near-identical
  model-construction functions the notebook defines.
- `training`: `TrainingConfig`/`train_model`, `tf.data` construction,
  deterministic seeding, and `.keras`/legacy `.h5` save/load.
- `evaluation`: accuracy/sensitivity/specificity/AUC from thresholded
  prediction probabilities, with explicit confusion-matrix edge-case
  handling.
- `gradcam`: a native `tf.GradientTape`-based 3D Grad-CAM, replacing the
  notebook's `tf-keras-vis` dependency (see `gradcam` module docstring for
  why).
- `regions`: AAL2 atlas region-importance ranking from Grad-CAM heatmaps,
  ported from `exploration.ipynb`.
"""
