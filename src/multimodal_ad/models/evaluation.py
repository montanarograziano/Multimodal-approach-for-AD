"""Evaluation metrics ported from `Training.ipynb`'s `get_matrix`/`sensitivity`/`specificity`.

**Retained**: 0.5 probability threshold (`tf.greater(m, 0.5)` in
`get_matrix`); `sensitivity = tp / (tp + fn)`, `specificity = tn / (tn +
fp)`, both from `sklearn.metrics.confusion_matrix`-style
`(tn, fp, fn, tp) = matrix.ravel()` ordering (note this is `sklearn`'s
`confusion_matrix` ravel order, distinct from the `tf.math.confusion_matrix`
ravel order used ad hoc elsewhere in the notebook, which is
`(tn, fp, tp, fn)` for binary labels — see inventory doc; this module
standardizes on `sklearn`'s convention throughout to avoid that
inconsistency).

**Documented divergence**: the notebook's `sensitivity`/`specificity`
divide unconditionally, raising `ZeroDivisionError`/emitting `nan` (via
NumPy) when a class is entirely absent from the evaluation batch. This
module returns `float("nan")` for that case explicitly (documented, not a
silent NumPy warning) rather than raising, since fold-level evaluation
batches can legitimately contain zero positives or zero negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

#: Retained from the legacy notebook's `get_matrix` (`tf.greater(m, 0.5)`).
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluationMetrics:
    """Accuracy/sensitivity/specificity/AUC plus the raw confusion-matrix counts."""

    accuracy: float
    sensitivity: float
    specificity: float
    auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def threshold_probabilities(
    probabilities: np.ndarray, *, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Binarize sigmoid output probabilities at `threshold` (legacy default `0.5`)."""
    probabilities = np.asarray(probabilities).reshape(-1)
    return (probabilities > threshold).astype(np.int64)


def sensitivity(tn: int, fp: int, fn: int, tp: int) -> float:
    """True positive rate. `nan` if there are no positives in the ground truth."""
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def specificity(tn: int, fp: int, fn: int, tp: int) -> float:
    """True negative rate. `nan` if there are no negatives in the ground truth."""
    denom = tn + fp
    return float(tn / denom) if denom > 0 else float("nan")


def evaluate_predictions(
    y_true: np.ndarray, probabilities: np.ndarray, *, threshold: float = DEFAULT_THRESHOLD
) -> EvaluationMetrics:
    """Compute accuracy/sensitivity/specificity/AUC from labels and predicted probabilities.

    `probabilities` are raw sigmoid outputs in `[0, 1]`; thresholding
    (`> threshold`) happens here, matching the legacy `get_matrix`. AUC is
    computed on the un-thresholded probabilities (as in the notebook's
    `roc_auc_score(ytest, predictions[:, 0])`), so it is independent of
    `threshold`.

    Raises `ValueError` if `y_true` contains anything other than exactly the
    labels `{0, 1}` (guards a mislabeled or empty evaluation batch, which
    the legacy notebook did not check for).
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    probabilities = np.asarray(probabilities).reshape(-1).astype(np.float64)
    if y_true.shape != probabilities.shape:
        raise ValueError(
            f"y_true and probabilities must have the same length, got {y_true.shape} "
            f"vs {probabilities.shape}"
        )
    labels_present = set(np.unique(y_true).tolist())
    if not labels_present <= {0, 1}:
        raise ValueError(f"y_true must contain only 0/1 labels, got {sorted(labels_present)}")

    y_pred = threshold_probabilities(probabilities, threshold=threshold)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

    accuracy = float((tn + tp) / len(y_true)) if len(y_true) > 0 else float("nan")
    auc = float(roc_auc_score(y_true, probabilities)) if len(labels_present) == 2 else float("nan")

    return EvaluationMetrics(
        accuracy=accuracy,
        sensitivity=sensitivity(tn, fp, fn, tp),
        specificity=specificity(tn, fp, fn, tp),
        auc=auc,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
    )
