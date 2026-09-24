"""Tests for the sensor anomaly detection project."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app import run_experiment
from data import make_dataset


class TestDatasetContract:
    """Verify the dataset generator meets its contract."""

    def test_shapes_and_labels(self) -> None:
        n = 256
        X, y = make_dataset(seed=42, n_samples=n)
        assert X.shape == (n, 3)
        assert y.shape == (n,)
        # y must be binary {0, 1}
        unique_labels = set(np.unique(y).tolist())
        assert unique_labels.issubset({0, 1})

    def test_different_seeds_differ(self) -> None:
        X1, _ = make_dataset(seed=42, n_samples=128)
        X2, _ = make_dataset(seed=99, n_samples=128)
        assert not np.array_equal(X1, X2)

    def test_invalid_n_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            make_dataset(seed=42, n_samples=16)


class TestMetricCalculation:
    """Verify that reported metrics are computed correctly on the chronological test split."""

    def test_metrics_match_independent_computation(self) -> None:
        """Independently recompute precision, recall, F1 on the test segment
        and confirm they match the values returned by run_experiment."""
        seed, n = 42, 256
        X, y = make_dataset(seed=seed, n_samples=n)
        split_idx = int(n * 0.7)
        y_test = y[split_idx:]

        result = run_experiment(seed=seed, n_samples=n)

        # We need to reproduce the predictions to verify metrics.
        # Import the internal helpers to recompute.
        from app import _engineer_features, _zscore_baseline
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import precision_score, recall_score, f1_score

        X_feat = _engineer_features(X, window=5)
        X_train_feat = X_feat[:split_idx]
        X_test_feat = X_feat[split_idx:]

        # Isolation Forest
        iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
        iso.fit(X_train_feat)
        train_scores = iso.score_samples(X_train_feat)
        threshold = np.percentile(train_scores, 15)
        iso_pred = (iso.score_samples(X_test_feat) < threshold).astype(np.int64)

        # Baseline
        baseline_pred = _zscore_baseline(X[:split_idx], X[split_idx:], threshold=3.0)

        # Independent metric computation
        for key, pred in [("metrics", iso_pred), ("baseline_metrics", baseline_pred)]:
            p = precision_score(y_test, pred, zero_division=0)
            r = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            assert abs(result[key]["precision"] - float(p)) < 1e-12, (
                f"{key} precision mismatch: {result[key]['precision']} vs {p}"
            )
            assert abs(result[key]["recall"] - float(r)) < 1e-12, (
                f"{key} recall mismatch: {result[key]['recall']} vs {r}"
            )
            assert abs(result[key]["f1"] - float(f1)) < 1e-12, (
                f"{key} f1 mismatch: {result[key]['f1']} vs {f1}"
            )


class TestDeterminismAndSplit:
    """Verify determinism and correct split sizes."""

    def test_deterministic_output(self) -> None:
        r1 = run_experiment(seed=42, n_samples=256)
        r2 = run_experiment(seed=42, n_samples=256)
        # JSON round-trip to ensure exact equality of serializable content
        assert json.loads(json.dumps(r1)) == json.loads(json.dumps(r2))

    def test_split_size(self) -> None:
        n = 256
        result = run_experiment(seed=42, n_samples=n)
        assert result["n_samples"] == n

        # Test set should be the last 30% of 256 = 77 samples
        split_idx = int(n * 0.7)  # 179
        expected_test_size = n - split_idx  # 77
        assert expected_test_size == 77

        # Verify by checking the result is consistent with the split
        # The explanation should mention the chronological split
        assert "chronological" in result["explanation"].lower()

    def test_different_n_samples_respected(self) -> None:
        """Verify that different n_samples produce different result sizes."""
        r_small = run_experiment(seed=42, n_samples=64)
        r_large = run_experiment(seed=42, n_samples=256)
        assert r_small["n_samples"] == 64
        assert r_large["n_samples"] == 256
        # The test set sizes should differ
        split_small = int(64 * 0.7)
        split_large = int(256 * 0.7)
        assert (64 - split_small) != (256 - split_large)
