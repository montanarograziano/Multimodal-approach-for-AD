# Legacy Notebooks: Behavior & Inventory

Phase 0 deliverable for the `refactor/foundation` migration. This document
characterizes the five legacy Colab notebooks as they exist today, **before**
any scientific code is ported to `src/multimodal_ad`. It distinguishes:

- **Paper semantics**: what the published method (Scientific Reports,
  [10.1038/s41598-024-56001-9](https://doi.org/10.1038/s41598-024-56001-9))
  describes or implies.
- **Notebook behavior**: what the code in this repo actually does, which
  sometimes diverges from the paper or from itself across cells.
- **Ambiguity**: places where intent cannot be recovered from the notebook
  alone (dead code, contradictory reassignments, undefined names, manual
  Colab/Drive steps).
- **Retained constants**: magic numbers/paths that must be preserved
  byte-for-byte during porting unless a future ADR changes them.

Notebooks are **not modified** in this PR. This is a read-only audit.

## Scope and general observations

All five notebooks (`Dataset_MRI.ipynb`, `Dataset_PET.ipynb`,
`Training.ipynb`, `Heatmaps.ipynb`, `exploration.ipynb`) were written as
linear, interactively-run Google Colab sessions against Google Drive-mounted
data (OASIS-3). They are **not idempotent, not deterministic across reruns**,
and contain:

- Hardcoded `DATA_DIR = Path("Your directory here")` placeholders (never a
  real path), meaning none of the notebooks currently execute end-to-end
  outside the original author's Colab environment.
- `!pip install`, `!mkdir`, `!ls`, `!cp`, `!rm` shell-outs mixed with Python,
  and `from google.colab import drive` (Colab-only import).
- `FIRST_RUN` boolean flags that branch between "compute and save" vs. "load
  cached artifact from disk", used inconsistently (some branches missing the
  `else`, some duplicated later).
- Repeated cells that recompute the same variables with slightly different
  logic (e.g., `get_merged()` is defined twice in `Training.ipynb` with
  different hidden layer widths: 128 then 4 units).
- References to undefined names in later cells (`F821` from Ruff: `xte`,
  `yte`, `hist`, `mri_metrics`, `pet_metrics`, `m`, `get_dataset`,
  `plot_folds`), meaning several cells cannot run standalone and their
  outputs were likely produced interactively with manual edits not saved back
  into the notebook.
- No tests, no CI, no dependency pinning beyond `pyproject.toml`/`uv.lock`
  added in the last commit (`fix: fix error in Dataset_MRI notebook and add
  pyprojct and uv.lock`), which lists packages but is not what any notebook
  actually imports (e.g., `mlflow`, `scikit-learn` are imported but were
  absent from `pyproject.toml`).

**Retained constant across all notebooks**: `RANDOM_SEED = 1234`.

---

## `Dataset_MRI.ipynb` — MRI dataset construction

**Paper semantics**: build a balanced, subject-deduplicated MRI dataset from
OASIS-3, label scans via nearest-in-time clinical diagnosis, augment the
minority (positive/AD) class via rotation/flip, resize to a common shape,
and produce stratified cross-validation folds.

**Notebook behavior**:

1. Loads five OASIS-3 CSVs (`subjects`, `clinical-data`, `pet`, `pup`,
   `mri`) and derives `Date`/`Subject`/`Tracer` columns via string-splitting
   the identifier columns (positional, format-dependent, not validated).
2. **Labeling**: a `dx1..dx5` differential-diagnosis regex classifies each
   diagnosis string as demented (`1`) or not (`0`):
   `^(AD dem|Vasc.*? dem|Frontotemporal dem|other mental retarAD demion|
   (Active )?DLBD|Active PSNP|Dementia)` (case-insensitive), OR
   `^uncertain.*?dem`. **Ambiguity**: `other mental retarAD demion` is almost
   certainly a corrupted/garbled regex fragment (typo `retarAD demion` does
   not correspond to any known OASIS diagnosis string) — the original intent
   is unrecoverable without the author. Comment says "we classify uncertain
   cases as sick", which is a **paper-relevant modeling decision**, not a
   bug, and must be preserved or explicitly revisited.
3. **Temporal label smoothing**: a hand-rolled forward/backward look-ahead
   algorithm (2 steps back, 2 steps forward) "corrects" isolated `False`
   readings surrounded by `True` diagnoses per subject, to counter visit-level
   noise in a progressive disease. **Ambiguity**: the loop mutates `prec`/
   `succ` windows with off-by-one-prone manual indexing (`index+1`, `index+2`,
   `index+3` checked against `.index` membership); behavior on the first/last
   1–2 rows per subject is under-specified and not covered by any assertion
   or test in the notebook.
4. **Nearest-visit labeling**: for each PUP (PET processing) row, finds the
   clinical visit with minimum `|Date - visit_date|` and assigns its
   (smoothed) `dementia` label. Ties are broken by `pandas` stable sort order
   (undocumented, not asserted).
5. **Subject dedup**: keeps only the most recent scan per subject
   (`drop_duplicates(..., keep="last")` after sorting by `Subject, Date`).
6. **Class balancing**:
   - Negative undersampling: `negative.sample(n=150, random_state=1234)`.
   - Positive augmentation: **43** randomly chosen positive MRI volumes
     (`random.sample(mri_images, k=43)`) are transformed via a weighted random
     choice of `rotate` (angle in `[-30, 30)`, weight 0.5), `flipv` (weight
     0.25), `fliph` (weight 0.25), with 1–5 operations chained
     (`random.randint(1, 5)`). Rotation fills exposed corners with the mean
     color of a `(5, 5)` background patch, using `scipy.ndimage.interpolation
     .rotate` (deprecated submodule; modern SciPy uses
     `scipy.ndimage.rotate`).
   - `n=150` negatives and `k=43` augmented positives are **retained
     constants** tuned to reach the paper's reported balanced counts, not
     rederivable from first principles here.
7. **Preprocessing** (`process_scan`):
   - `normalize()`: **min-max scaling to `[0, 1]`**, `float32`, based on the
     volume's own min/max (not a fixed intensity range, not per-atlas or
     population normalization).
   - Handles 4D volumes (some MRI/PET files have a 4th "echo"/frame axis) by
     averaging over axis 3 (`np.mean(volume, axis=3)`) before normalizing.
   - `find_brain_bounding_box()`: per-slice Otsu thresholding
     (`cv2.GaussianBlur((51,51), sigma=50)` → normalize to `uint8` →
     `cv2.threshold(THRESH_BINARY+THRESH_OTSU)` → `cv2.medianBlur(51)` →
     largest contour's bounding rect), padded by 10px, then unioned
     (max/min) across all 50 central slices to get one box for the whole
     volume. **Ambiguity**: `cv2.findContours(...)[0]` is used, which assumes
     at least one contour is found; a scan with a fully empty/degenerate
     mask will raise `IndexError` (no fallback).
   - `resize_to_input_shape()`: extracts the **50 middle frames** along the
     depth axis, square-crops around the unioned bounding box (padding the
     shorter dimension), then `cv2.resize` to **128×128** per frame. Final
     shape: **(128, 128, 50)**. This is the paper's reported input shape.
   - PET's equivalent bounding-box function in `Dataset_PET.ipynb` uses
     **different Gaussian/threshold parameters**
     (`GaussianBlur((13,13), sigma=150)`, `medianBlur(5)`) than MRI's
     (`GaussianBlur((51,51), sigma=50)`, `medianBlur(51)`) — this is a
     **deliberate, modality-specific tuning difference**, not a copy-paste
     bug, and must be preserved per-modality.
8. **CV folds**: multiple, inconsistent fold-generation strategies appear
   across cells: a 10×10 `RepeatedStratifiedKFold` over *subjects* (100
   folds, each replicating each subject's label 50× to match the per-frame
   sample count), and later a single 10-fold `StratifiedKFold` directly over
   already-loaded frame-level arrays. **Ambiguity**: it is not clear from the
   notebook alone which fold scheme was used for the paper's final reported
   numbers; both code paths exist with hardcoded output directories
   (`3d-folds/20`, `subjects/fold-N`, `folds/fold-N`) that were never
   reconciled into one canonical pipeline. This must be resolved with the
   paper authors before porting Phase 2 training code.
9. Hardcoded exclusion list of 9 specific `MRId` values (e.g.
   `'OAS30031_MR_d0236'`) filtered out of the training/test split with no
   comment explaining why (likely corrupted files found during a prior run;
   **retained as-is**, flagged as an ambiguity).
10. Train/test subject split is **positional, not random**: `sample[:75]`
    (train positive), `sample[75:105]` (test positive), `sample[105:223]`
    (train negative), `sample[223:]` (test negative), applied after sorting.
    This is a **fixed 75/30 pos, 118/? neg split** baked into slice indices
    rather than a parameterized `train_test_split` call — a retained
    constant, fragile to any upstream change in `sampled`'s row count/order.

**Retained constants (MRI)**: `RANDOM_SEED=1234`; negative sample
`n=150`; positive augmentation `k=43`; augmentation angle range
`[-30, 30)`; augmentation op weights `[0.5, 0.25, 0.25]`; augmentation op
count `randint(1,5)`; bounding box padding `10px`; MRI Otsu params
`GaussianBlur((51,51), sigma=50)` / `medianBlur(51)`; output shape
`(128, 128, 50)`; central-frame extraction of **50** frames (later cells also
reference 20- and 30-frame variants — see Ambiguity above); positional
split indices `[:75]/[75:105]/[105:223]/[223:]`; 9 hardcoded excluded
`MRId`s.

---

## `Dataset_PET.ipynb` — PET dataset construction

**Paper semantics**: mirror the MRI pipeline for amyloid PET (PIB/AV45
tracers), producing a matched, labeled, balanced, resized dataset.

**Notebook behavior**: structurally near-identical to `Dataset_MRI.ipynb`
(same labeling regex, same temporal smoothing algorithm, same
`RANDOM_SEED`, same `n=150` negative sample, same `k=43` positive
augmentation) with these PET-specific differences:

- Paths use `.img`/`.hdr` (Analyze format) for raw scans, `.4dfp.nii` for
  augmented/modified outputs.
- `find_brain_bounding_box` uses PET-tuned parameters:
  `GaussianBlur((13,13), sigma=150)`, `medianBlur(5)` — see MRI section for
  contrast. **Retained, modality-specific.**
- **Ambiguity — inconsistent frame counts**: the notebook alternates between
  processing **50**, **30**, and **20** central frames in different cells
  (`n_frames=50` in `resize_to_input_shape`, but later cells filter on
  `shape[-1] == 30` or `== 20` and save to differently-named files
  `20-xtrain.npy`, `30-xtrain.npy`, `50-xtrain.npy`). This strongly suggests
  the author was experimenting with different depth truncations across
  training runs; **which one produced the paper's final reported metrics is
  not determinable from the notebook alone** and needs authors'
  confirmation before Phase 2 ports a single canonical value.
- A regex-based path remapping
  (`(^OAS\d+\_)(AV45|PIB)_PUPTIMECOURSE_(d\d+)` →
  `\1\2_\3n_moco`) reconciles two different naming conventions for the same
  scan between the `subjects_x*.npy` manifest and the actual PUP file
  tree — fragile, undocumented, PET-specific.
- 7 hardcoded row indices (`[55, 73, 80, 150, 180, 193, 203]`) are deleted
  from `ytrain` to align with images dropped during processing (broken
  files). No corresponding comment; **ambiguity**, retained as-is.
- `sample.sort_values(["Subject", "Date"])` +
  `drop_duplicates(subset=["Subject"], keep="last")` is applied a **second
  time** in this notebook (already done once when producing
  `new_sample.csv`), which is redundant but harmless given already-deduped
  input — not a behavior change, just noise.
- A `train_test_split(x, y, test_size=0.2)` validation-split cell exists but
  operates on undefined `x`/`y` (never assigned in a preceding visible cell)
  — **dead/broken cell**, not part of the effective pipeline as saved.

**Retained constants (PET)**: same as MRI section plus PET Otsu params
`GaussianBlur((13,13), sigma=150)` / `medianBlur(5)`; 7 hardcoded excluded
`ytrain` indices; regex path-remap pattern above.

---

## `Training.ipynb` — model architectures & training

**Paper semantics**: a 3D CNN (based on Zunair et al.,
[arXiv:2007.13224](https://arxiv.org/abs/2007.13224)) trained separately on
MRI and PET, then combined via late-fusion transfer learning and a merged
dual-input model, evaluated via repeated k-fold cross-validation with
confusion-matrix metrics logged to MLflow (via DagsHub).

**Notebook behavior**:

- **Base 3D CNN** (`get_3d_model`, input `(128, 128, 50, 1)`): four
  `Conv3D → MaxPool3D(2) → BatchNorm` blocks with filter widths
  **64, 64, 128, 256** (kernel size 3, ReLU), then
  `GlobalAveragePooling3D → Dense(512, relu) → Dropout(0.3) →
  Dense(1, sigmoid)`. This is the paper's core architecture.
- **Frozen variant** (`get_3d_frozen`): identical architecture with every
  `Conv3D`/`BatchNormalization` layer's `trainable=False`, used for transfer
  learning (only the dense head trains). Named variants `get_3d_x`/`get_3d_y`
  add explicit `name="input_x"/"input_y"` Input layers and named dropout
  layers, for later merging.
- **Training loop** (`train_3d_model`): `binary_crossentropy` loss,
  `Adam` with `ExponentialDecay(initial_lr=5e-5, decay_steps=100000,
  decay_rate=0.96, staircase=True)`. `5e-5` is cited from
  [doi:10.3938/jkps.75.597](https://doi.org/10.3938/jkps.75.597), i.e. a
  **paper-justified constant**, not arbitrary. `EarlyStopping(monitor=
  "val_acc", patience=35, min_delta=0.001, baseline=0.60)` — note
  **`baseline=0.60`** means early stopping's patience/restore logic only
  engages once validation accuracy has ever exceeded 60%; runs that never
  cross that threshold train for the full `epochs` budget (default
  `10000`). `ModelCheckpoint(monitor="val_acc", save_best_only=True)`.
  `batch_size=32` default, overridden to `4`–`8` in several transfer-learning
  calls (data-volume dependent, since 3D volumes are large).
- **Transfer learning variants** (MRI→PET and PET→MRI, with 0/1/2 unfrozen
  conv layers): `tuning_3d()` reloads a frozen model, loads pretrained
  weights from the *other* modality's best checkpoint, and fine-tunes. The
  early-stopping `patience` is inconsistently **35** in one cell and **20**
  in the near-duplicate `tuning_3d` defined right after it — **ambiguity**:
  two definitions of the same function with different defaults, the second
  silently shadows the first, and it is not clear which was in effect for
  which reported experiment.
- **Merged (late-fusion) model** (`get_merged`): loads both single-modality
  models' best weights, strips each model's last 3 layers, concatenates
  their penultimate `GlobalAveragePooling3D` outputs, and adds
  `Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)`. **Ambiguity**: a
  **second, later definition of `get_merged()`** in the same notebook
  changes the fusion head to `Dense(4, relu)` and reassigns `z =
  Dropout(0.3)(merged)` (dropping the `Dense(4, relu)` output entirely, an
  apparent bug — the `Dense(4, ...)` layer becomes dead/unused since `z` is
  immediately overwritten). Because the second definition executes later
  and shadows the first in a live notebook session, **the actual head
  topology used for the paper's merged-model numbers cannot be determined
  from static inspection alone.**
- **Combined 10-fold CV**: trains the base 3D model directly on a
  concatenated MRI+PET dataset (not a fusion architecture, just pooled
  data) across `StratifiedKFold(n_splits=10, random_state=1234,
  shuffle=True)`; variable name `kFold` (flagged by Ruff `N816` — mixedCase
  — harmless).
- Multiple cells (`get_matrix`, `plot_folds`, `plot_results`,
  `sensitivity`) reference each other or globals (`xte`, `yte`, `hist`,
  `mri_metrics`, `pet_metrics`, `m`) that are **not defined in the visible
  notebook state** — these were almost certainly produced by earlier,
  now-deleted or reordered cells in the author's interactive session, and
  their exact values/shapes cannot be reconstructed from the file as
  committed.
- MLflow tracking is wired to a hardcoded DagsHub project
  (`input()`-prompted username/project at runtime) — **not reproducible in
  CI or any non-interactive environment**, and will need to be replaced
  with local or self-hosted tracking (or removed) when ported.

**Retained constants (Training)**: architecture filter widths
`[64, 64, 128, 256]`, kernel size `3`, pool size `2`, dropout `0.3`, dense
head `512` units; learning rate `5e-5` (cited, paper-justified); LR decay
`decay_steps=100000, decay_rate=0.96, staircase=True`; early stopping
`monitor="val_acc", min_delta=0.001, baseline=0.60`, patience **35** (primary
path) vs. **20** (shadowed duplicate, ambiguous); input shape
`(128, 128, 50, 1)`.

---

## `Heatmaps.ipynb` — Grad-CAM interpretability

**Paper semantics**: generate Grad-CAM saliency maps per class (positive/
negative) to visualize which brain regions drive the model's predictions,
averaged across samples for a population-level heatmap.

**Notebook behavior**:

- A hand-written `make_gradcam_heatmap()` (2D, single-layer, adapted from the
  standard Keras Grad-CAM recipe) is defined but appears **unused** in the
  final PET/MRI sections — the notebook switches to the
  [`tf-keras-vis`](https://github.com/keisen/tf-keras-vis) library's
  `Gradcam` class instead (`penultimate_layer=-7`, `CategoricalScore([0]*5)`
  — the score list length **5** does not match the actual batch size used in
  the loop below it, `i:i+5`; this is a coincidental match to the batch
  chunk size, not a fixed "5 classes" semantic — **ambiguity/fragility**, not
  a bug given how it's called, but easy to break if batch chunking changes).
- `tf.config.run_functions_eagerly(True)` is required for Grad-CAM gradient
  computation against the graph-mode model — a **retained requirement** for
  any TF/Keras Grad-CAM port.
- `model.layers[-1].activation = None` unwraps the sigmoid before computing
  gradients (standard Grad-CAM practice, to use raw logits) — **retained**.
- Layer name **`conv2d_3`** is hardcoded for the (2D, seemingly PET-only)
  path and **`conv3d_3`** for MRI, both **fragile Keras auto-naming**
  (depends on model construction order/session state — will silently break
  or point at the wrong layer if the model is rebuilt with different code
  around it). Must be replaced with explicit `name=` layer arguments when
  ported.
- Post-hoc masking (`get_masked`) re-applies an Otsu brain mask
  (`GaussianBlur((21,21), sigma=150)`, `medianBlur(1)`, note **not** the
  same params as the dataset notebooks' brain-bounding-box functions — a
  **third, independent tuning** of the same underlying technique) to zero
  out heatmap signal outside the skull.
- `overlay()` = plain `np.mean(arr, axis=0, dtype=np.float64)` across the
  sample axis — the population-level heatmap is a simple voxel-wise mean,
  no weighting by prediction confidence.
- Several cells reference precomputed `.npy` heatmap artifacts
  (`pos_mean_mri.npy`, `neg_mean_pet.npy`, etc.) that are **checked into the
  repo root** (see "Repo-root data artifacts" below) rather than regenerated
  in-notebook — meaning the notebook as committed is not fully
  self-contained even with real data.

**Retained constants (Heatmaps)**: `penultimate_layer=-7` for `tf-keras-vis`
Gradcam; masking Otsu params `GaussianBlur((21,21), sigma=150)`,
`medianBlur(1)`; `run_functions_eagerly(True)` requirement.

---

## `exploration.ipynb` — anatomical zone ranking

**Paper semantics**: rank AAL2 atlas brain regions by mean Grad-CAM
importance, separately for MRI/PET and positive/negative class, to relate
model attention to known AD-affected regions.

**Notebook behavior**:

- Loads `atlas.nii.gz` (AAL2 atlas, referenced via a NeuroVault URL in the
  markdown; at the time of this audit the file was **not present in this
  repo** and had to be sourced separately — since resolved, see the module
  map above and `models.regions.load_atlas`) and `AAL2_Atlas_Labels.csv`
  (checked into repo root, maps region name → intensity value).
- `fix_heat_dim()` / `fix_atlas_dim()`: pad the (128,128,50) heatmap and the
  atlas volume into a common **(128, 128, 128)** frame, using **hardcoded
  offset slices**: heatmap placed at `[:, :, 39:89]` (50 slices centered in
  128), atlas placed at `[19:110, 10:119, 19:110]`. **Ambiguity**: these
  offsets assume specific, unverified spatial alignment between the AAL2
  atlas's native resolution/origin and the pipeline's resized MRI/PET
  volumes — there is no registration step shown; alignment correctness
  cannot be verified from the notebook alone and should be treated as a
  known limitation, not a validated coregistration.
- For each atlas region, computes masked mean/count/sum over the padded
  heatmaps, separately for {MRI, PET} × {positive, negative}, and writes
  `heatmap_importance.csv`. **Denominator note**: `mean` is `masked.mean()`
  on the array *after* zeroing outside the region but *before* slicing to
  the region, so it divides by the full (128, 128, 128) common-frame voxel
  count, not the region's own voxel count — a small region's `mean` is
  diluted by a mostly-zero denominator, not a smaller per-voxel signal. The
  port in `multimodal_ad.models.regions.rank_regions` keeps this exact
  formula as `f"{name} mean"` (for reproducing the notebook's/paper's
  numbers) and adds a size-comparable `f"{name} region mean"` (sum divided
  by the region's own voxel count) for new analysis.
- `!pipenv install` in the first cell is a leftover from a different,
  unrelated dependency manager than the rest of the project (which now uses
  `uv`) — **dead, must not be treated as a dependency source of truth**.
- Uses `pandas` `.head()` (top-5, ascending) for "best" attention regions —
  this treats **lowest** mean value as most important because it sorts
  ascending, which is directionally sensible only under the notebook's own
  implicit convention that darker/lower-signal Grad-CAM values in a jet
  colormap denote background; there is no comment confirming this reading,
  and Phase 2 porting must confirm the sign/convention with the paper before
  reusing this ranking logic.

**Retained constants (exploration)**: padding offsets `heat[:, :, 39:89]`,
`atlas[19:110, 10:119, 19:110]`; atlas region label CSV format
(`name, intensity`, no header).

---

## Repo-root data artifacts (not notebooks, but load-bearing)

Several binary artifacts live at the repository root and are referenced by
the notebooks above. They are historical outputs and are **left untouched**
in this PR:

| File | Referenced by | Notes |
| --- | --- | --- |
| `mri1.nii` | `images/3D Brain Plot.ipynb` | Loaded via `nib.load()` and used as the affine/spatial reference volume for rendering the 3D Grad-CAM figures. |
| `resized_pos_mri.npy`, `resized_neg_mri.npy` | `images/3D Brain Plot.ipynb`, `exploration.ipynb` | Derived Grad-CAM artifacts: resized single-volume MRI heatmaps. Used in `images/3D Brain Plot.ipynb` for the positive-class preview slice (`plt.imshow(resized_pos_mri[:, :, 10])`) and in `exploration.ipynb` as the MRI heatmaps (`heat_mri_pos`/`heat_mri_neg`) driving AAL2 region ranking. |
| `pos_mean_mri.npy`, `pos_mean_pet.npy` | `Heatmaps.ipynb`, `exploration.ipynb` | Precomputed mean Grad-CAM heatmaps. `neg_mean_pet.npy` and `pos_mri_heat.npy`/`neg_mri_heat.npy` are referenced by `exploration.ipynb`/`Heatmaps.ipynb` but **not present** in the repo — an existing gap, not introduced by this PR. |
| `AAL2_Atlas_Labels.csv` | `exploration.ipynb` | Atlas region name → intensity mapping. |
| `samples/mri-sample.png`, `samples/pet-sample.png` | README (implicitly, via `images/3D Brain Plot.ipynb`) | Figure assets for the paper. |

## `images/` sub-project (moved in the Phase 3 notebooks PR)

`images/3D Brain Plot.ipynb` used to have its own, separate Poetry project
(`images/pyproject.toml` + `images/poetry.lock`, `dementiadetection`
0.1.0, Python `^3.9`, depends on `nilearn`/`numpy`/`matplotlib`/`notebook`/
`opencv-python`/`pandas`) used only to generate the paper's glass-brain
figures. An earlier revision of this document deferred touching it
(isolated lockfile, no risk to the root `uv` environment). The Phase 3
notebooks PR resolves that deferral: the notebook is now preserved
byte-for-byte at `notebooks/legacy/3d-brain-plot.ipynb` alongside the other
four legacy notebooks, and `images/pyproject.toml`/`images/poetry.lock`
are **removed** (not just deferred), since the Poetry sub-project existed
only to pin dependencies for that one notebook: with the notebook moved
out of `images/`, the sub-project has no remaining purpose and keeping a
second, unmaintained Python dependency toolchain (Poetry, pinned to Python
`^3.9`, unrelated to this project's `uv`/3.13 toolchain) around for no
referenced file would be needless upkeep, not a preserved artifact. The
`images/` directory itself is gone (it held only the notebook and the
Poetry files); `samples/*.png` (the rendered figure PNGs `images/3D Brain
Plot.ipynb` produces) are untouched at the repository root, since those
are the actual image assets, not the notebook/toolchain that generates
them.

## Summary: what must be resolved before Phase 2 (scientific code porting)

1. **Confirm final frame-depth used for paper results**: 20, 30, or 50
   central frames (both MRI and PET notebooks contain all three, produced at
   different times).
2. **Confirm final merged-model fusion head**: `Dense(128, relu)` vs.
   `Dense(4, relu)` (with the `Dense(4,...)` output apparently discarded) —
   two conflicting `get_merged()` definitions exist.
3. **Confirm early-stopping patience** for transfer-learning fine-tuning: 35
   vs. 20 (two conflicting `tuning_3d()` definitions).
4. **Confirm the "uncertain" and garbled-regex diagnosis mapping** in the
   dementia-labeling regex against the OASIS-3 codebook, especially the
   apparently corrupted `other mental retarAD demion` fragment.
5. **Confirm CV fold scheme** used for reported metrics: repeated 10×10
   subject-level `RepeatedStratifiedKFold` vs. single 10-fold frame-level
   `StratifiedKFold`.
6. **Source the AAL2 atlas file** (`atlas.nii.gz`) separately; it was not
   checked into the repo at the time of this audit. **Resolved**: the file
   is now bundled at the repo root, see `models.regions.load_atlas`.
7. **Replace Colab/Drive/MLflow-via-DagsHub-with-`input()`** assumptions with
   a reproducible, non-interactive data/experiment-tracking story.

None of the above block Phase 0/1 (this PR). They are the explicit input to
scoping Phase 2's scientific code porting plan.
