"""Multivariate Sensor Anomaly Detection via Isolation Forest.

Detects anomalies in synthetic 3-channel sensor time series using
feature-engineered Isolation Forest versus a per-feature z-score
threshold baseline on a chronological train/test split.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from data import make_dataset


def _engineer_features(X: np.ndarray, window: int = 5) -> np.ndarray:
    """Engineer rolling mean, rolling std, and first-difference features.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 3)
        Raw sensor data.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray of shape (n_samples, 9)
        Features: [rolling_mean(3), rolling_std(3), first_diff(3)]
    """
    n = X.shape[0]
    # Rolling mean
    roll_mean = np.zeros_like(X)
    for i in range(n):
        start = max(0, i - window + 1)
        roll_mean[i] = X[start : i + 1].mean(axis=0)

    # Rolling std (use ddof=0 for population std to avoid NaN at i=0)
    roll_std = np.zeros_like(X)
    for i in range(n):
        start = max(0, i - window + 1)
        roll_std[i] = X[start : i + 1].std(axis=0, ddof=0)

    # First difference (0 for the first sample)
    first_diff = np.zeros_like(X)
    if n > 1:
        first_diff[1:] = np.diff(X, axis=0)

    return np.column_stack([roll_mean, roll_std, first_diff])


def _zscore_baseline(
    X_train: np.ndarray, X_test: np.ndarray, threshold: float = 3.0
) -> np.ndarray:
    """Per-feature z-score anomaly detection.

    Statistics are computed only from the training segment to avoid leakage.

    Parameters
    ----------
    X_train : np.ndarray
        Training data for computing mean and std.
    X_test : np.ndarray
        Test data to score.
    threshold : float
        Z-score threshold for anomaly classification.

    Returns
    -------
    np.ndarray of shape (n_test,)
        Binary predictions: 0 = normal, 1 = anomaly.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    # Avoid division by zero
    std = np.where(std < 1e-12, 1.0, std)

    z_scores = np.abs((X_test - mean) / std)
    # A sample is anomalous if any feature exceeds the threshold
    predictions = (z_scores.max(axis=1) > threshold).astype(np.int64)
    return predictions


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """Run the anomaly detection experiment.

    Parameters
    ----------
    seed : int
        Random seed for dataset generation.
    n_samples : int
        Number of time steps. Must be >= 32.

    Returns
    -------
    dict
        JSON-serializable dict with keys:
        - n_samples: int
        - metrics: dict[str, float] with precision, recall, f1
        - baseline_metrics: dict[str, float] with precision, recall, f1
        - explanation: str
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    # Chronological 70/30 split (no shuffling)
    split_idx = int(n_samples * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Feature engineering is causal (uses only past and current values),
    # so engineering on the full sequence does not leak future information
    # into the training segment.
    X_feat = _engineer_features(X, window=5)
    X_train_feat = X_feat[:split_idx]
    X_test_feat = X_feat[split_idx:]

    # --- Isolation Forest ---
    iso = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=seed,
    )
    iso.fit(X_train_feat)
    # score_samples returns negative anomaly scores (lower = more anomalous)
    train_scores = iso.score_samples(X_train_feat)
    # Set threshold at the 15th percentile of training scores
    # (i.e., 15% of training points are considered anomalous)
    threshold = np.percentile(train_scores, 15)
    iso_predictions = (iso.score_samples(X_test_feat) < threshold).astype(np.int64)

    # --- Z-score Baseline ---
    baseline_predictions = _zscore_baseline(X_train, X_test, threshold=3.0)

    # --- Metrics ---
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }

    metrics = _compute_metrics(y_test, iso_predictions)
    baseline_metrics = _compute_metrics(y_test, baseline_predictions)

    explanation = (
        "Chronological 70/30 split: first 70% of samples form the training "
        "segment, last 30% form the held-out test segment. No shuffling is "
        "performed to respect temporal ordering and prevent leakage. "
        "Feature engineering (rolling mean, rolling std, first difference "
        "with window=5) is causal, using only past and current values. "
        "Isolation Forest is trained on training features; the anomaly "
        "threshold is set at the 15th percentile of training scores "
        "(equivalent to 85th percentile in anomaly-score space). "
        "The z-score baseline computes per-feature mean and std from the "
        "training segment only and flags a sample as anomalous if any "
        "feature's |z-score| exceeds 3.0."
    )

    return {
        "n_samples": int(n_samples),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "explanation": explanation,
    }
