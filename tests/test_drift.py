import numpy as np
import pandas as pd

from core.drift import psi_categorical, psi_numeric, status_from


def test_no_drift_same_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(0, 1, 2000))
    assert psi_numeric(a, b) < 0.10


def test_drift_detected_when_shifted():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(2, 1, 2000))
    assert psi_numeric(a, b) > 0.25


def test_categorical_drift():
    a = pd.Series(["x"] * 900 + ["y"] * 100)
    b = pd.Series(["x"] * 300 + ["y"] * 700)
    assert psi_categorical(a, b) > 0.25


def test_tiny_samples_return_none():
    assert psi_numeric(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) is None


def test_status_thresholds():
    assert status_from(0.05) == "stable"
    assert status_from(0.15) == "moderate"
    assert status_from(0.40) == "significant"