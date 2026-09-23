"""Tests for threshold selection and calibration application
(src/train_model.py)."""
import numpy as np
import pytest

from train_model import apply_calibration, youden_threshold


class _StubCalibrator:
    """Minimal stand-in for a fitted LogisticRegression calibrator: always
    returns a fixed probability, so tests can check apply_calibration's
    routing logic without needing a real fit.
    """
    def __init__(self, fixed_value: float):
        self.fixed_value = fixed_value

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.fixed_value), np.full(n, self.fixed_value)])


def test_youden_threshold_perfect_separation():
    # scores perfectly separate the two classes at 0.5
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = youden_threshold(y, p)
    assert 0.3 < thr <= 0.7


def test_youden_threshold_is_clipped_to_valid_range():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    thr = youden_threshold(y, p)
    assert 0.05 <= thr <= 0.95


def test_apply_calibration_routes_by_state():
    p = np.array([0.4, 0.4, 0.4, 0.4])
    states = np.array(["CA", "CA", "NY", "NY"])
    calibrators = {"CA": _StubCalibrator(0.9), "NY": _StubCalibrator(0.1)}
    out = apply_calibration(p, states, calibrators)
    assert out.shape == p.shape
    assert np.allclose(out[states == "CA"], 0.9)
    assert np.allclose(out[states == "NY"], 0.1)


def test_apply_calibration_leaves_uncalibrated_states_unchanged():
    p = np.array([0.55, 0.66])
    states = np.array(["CA", "NY"])
    calibrators = {"CA": None, "NY": _StubCalibrator(0.8)}
    out = apply_calibration(p, states, calibrators)
    assert out[0] == pytest.approx(0.55)   # CA has no calibrator -> passthrough
    assert out[1] == pytest.approx(0.8)    # NY calibrated


def test_apply_calibration_output_is_valid_probability_range():
    p = np.array([0.2, 0.5, 0.8])
    states = np.array(["CA", "CA", "CA"])
    calibrators = {"CA": _StubCalibrator(0.65)}
    out = apply_calibration(p, states, calibrators)
    assert ((out >= 0) & (out <= 1)).all()
