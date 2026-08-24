"""Tests for `multimodal_ad.models.evaluation` (no TensorFlow needed)."""

import numpy as np
import pytest

from multimodal_ad.models.evaluation import (
    evaluate_predictions,
    sensitivity,
    specificity,
    threshold_probabilities,
)


def test_threshold_probabilities_uses_strict_greater_than_half() -> None:
    probs = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    assert threshold_probabilities(probs).tolist() == [0, 0, 0, 1, 1]


def test_sensitivity_and_specificity_reference_case() -> None:
    # tn=3, fp=1, fn=2, tp=4
    assert sensitivity(tn=3, fp=1, fn=2, tp=4) == pytest.approx(4 / 6)
    assert specificity(tn=3, fp=1, fn=2, tp=4) == pytest.approx(3 / 4)


def test_sensitivity_is_nan_when_no_positives() -> None:
    assert np.isnan(sensitivity(tn=5, fp=0, fn=0, tp=0))


def test_specificity_is_nan_when_no_negatives() -> None:
    assert np.isnan(specificity(tn=0, fp=0, fn=0, tp=5))


def test_evaluate_predictions_reference_case() -> None:
    # 4 samples: perfect predictions except one false positive.
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.9, 0.8])
    metrics = evaluate_predictions(y_true, probabilities)

    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.true_positives == 2
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.sensitivity == pytest.approx(1.0)
    assert metrics.specificity == pytest.approx(0.5)
    assert 0.0 <= metrics.auc <= 1.0


def test_evaluate_predictions_perfect_separation_has_auc_one() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    metrics = evaluate_predictions(y_true, probabilities)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.auc == pytest.approx(1.0)


def test_evaluate_predictions_single_class_batch_returns_nan_auc_and_metric() -> None:
    """A fold with only negatives: AUC undefined, sensitivity undefined (no positives)."""
    y_true = np.array([0, 0, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.6, 0.05])
    metrics = evaluate_predictions(y_true, probabilities)
    assert np.isnan(metrics.auc)
    assert np.isnan(metrics.sensitivity)
    assert metrics.specificity == pytest.approx(0.75)  # 3 correct out of 4 negatives
    assert metrics.accuracy == pytest.approx(0.75)


def test_evaluate_predictions_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions(np.array([0, 1]), np.array([0.5]))


def test_evaluate_predictions_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="0/1 labels"):
        evaluate_predictions(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))
