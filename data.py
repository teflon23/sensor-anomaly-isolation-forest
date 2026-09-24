"""Synthetic 3-channel sensor dataset generator.

Generates a multivariate sensor time series with injected anomalies.
Each sample is a row of 3 sensor channels (temperature, pressure, vibration).
Anomalies are injected as multi-channel deviations to simulate realistic
sensor fault events.
"""

from __future__ import annotations

import numpy as np


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 3-channel sensor dataset with anomalies.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of time steps. Must be >= 32.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 3)
        Three sensor channels: temperature, pressure, vibration.
    y : np.ndarray of shape (n_samples,)
        Binary labels: 0 = normal, 1 = anomaly.

    Raises
    ------
    ValueError
        If n_samples < 32.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)

    # --- Normal signal generation (smooth, correlated channels) ---
    t = np.arange(n_samples, dtype=np.float64)

    # Temperature: slow sine + small noise
    temp = 20.0 + 3.0 * np.sin(2.0 * np.pi * t / 100.0) + rng.normal(0, 0.3, n_samples)

    # Pressure: slow cosine + small noise, correlated with temperature
    pressure = 101.3 + 2.0 * np.cos(2.0 * np.pi * t / 120.0) + 0.5 * (temp - 20.0) / 3.0 + rng.normal(0, 0.2, n_samples)

    # Vibration: low-amplitude noise + small periodic component
    vibration = 0.5 + 0.1 * np.sin(2.0 * np.pi * t / 20.0) + rng.normal(0, 0.05, n_samples)

    X = np.column_stack([temp, pressure, vibration])

    # --- Anomaly injection ---
    # Anomaly rate ~15%, injected as multi-channel spikes
    n_anomalies = max(3, int(0.15 * n_samples))
    anomaly_indices = rng.choice(n_samples, size=n_anomalies, replace=False)
    anomaly_indices = np.sort(anomaly_indices)

    y = np.zeros(n_samples, dtype=np.int64)
    y[anomaly_indices] = 1

    for idx in anomaly_indices:
        # Each anomaly affects 2-3 channels with a spike
        n_channels_affected = rng.integers(2, 4)  # 2 or 3
        channels = rng.choice(3, size=n_channels_affected, replace=False)
        for ch in channels:
            direction = rng.choice([-1, 1])
            magnitude = rng.uniform(3.0, 6.0)
            X[idx, ch] += direction * magnitude

    return X, y
