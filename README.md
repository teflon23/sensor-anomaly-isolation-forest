# Multivariate Sensor Anomaly Detection via Isolation Forest

## Problem

Detect anomalies in synthetic 3-channel sensor time series using feature-engineered Isolation Forest versus a per-feature z-score threshold baseline on a chronological train/test split.

## Actual Implementation

This project implements a batch time-series anomaly detection task requiring temporal feature engineering and a non-linear ensemble detector, validated on a strict chronological split to prevent future-into-past leakage.

### Algorithm

1. **Feature Engineering**: Engineer rolling mean (w=5), rolling std (w=5), and first-difference features from 3 raw sensor channels.
2. **Isolation Forest**: Train Isolation Forest (n_estimators=100, contamination='auto', random_state=seed) on chronological train (first 70%).
3. **Threshold**: Set anomaly threshold at the 15th-percentile of training scores (equivalent to 85th percentile in anomaly-score space).

### Baseline

Per-feature z-score with |z|>3 threshold, statistics computed only from the training segment. A sample is flagged as anomalous if any feature's z-score exceeds the threshold.

### Validation

Chronological 70/30 split (no shuffling, no overlap) to respect temporal ordering and prevent leakage.

## Architecture

- `data.py`: Synthetic 3-channel sensor dataset generator with injected anomalies.
- `app.py`: Feature engineering, Isolation Forest training, z-score baseline, and metric computation.
- `test_project.py`: Pytest test cases for dataset contract, algorithm vs baseline, and determinism.
- `main.py`: Host-supplied entry point for running experiments.
- `requirements.txt`: Pinned dependencies.

## Synthetic Dataset Assumptions

- 3 sensor channels: temperature, pressure, vibration.
- Normal signals: smooth, correlated channels with small noise.
- Anomalies: ~15% rate, injected as multi-channel spikes (2-3 channels affected per anomaly).
- Limitations: Synthetic data does not capture real-world sensor drift, missing values, or complex fault patterns.

## Algorithm vs Baseline

- **Isolation Forest**: Non-linear ensemble detector trained on engineered features.
- **Z-score Baseline**: Linear per-feature thresholding.
- **Note**: Algorithm-vs-baseline performance is a measured result, not a universal correctness invariant. The baseline may win. Refer to `validation_report.json` and `example_results.json` for measured output.

## Metrics Direction

- **Precision, Recall, F1**: All higher is better.
- **Evaluation**: Binary anomaly labels in the held-out test segment.

## Limitations

- Synthetic data only; not representative of real-world sensor systems.
- Fixed 70/30 chronological split; no cross-validation.
- Isolation Forest threshold is set based on training scores, which may not generalize optimally.
- Z-score baseline assumes feature independence and normality, which may not hold.
- No handling of missing values, sensor drift, or complex temporal patterns.

## Reproducibility

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run Experiment

```bash
python main.py --seed 42 --n-samples 256 --output results.json
```

### Run Tests

```bash
python -m pytest -q
```

### Python Version

Python 3.11-3.13.

### No Production Readiness

This project is for educational purposes only and is not intended for production use.

### No Invented Numerical Results

Refer to `validation_report.json` and `example_results.json` for measured output.

### No CI Workflow or Hosted Service

There is no CI workflow or hosted service. Do not invent badges or URLs.

### Limits of Automated Tests

Automated tests verify dataset contract, algorithm vs baseline, and determinism. They do not guarantee real-world performance or robustness to unseen data patterns.

## Recorded automated validation

Host contract tests and project tests passed (11 tests, 0 skipped). Demo completed on Python 3.13.15. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
