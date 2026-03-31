# Loss Interpretation & Baseline Rationale

## Context

We are training a seq2seq model to impute (fill) missing values in electricity
consumption time series. The data is **standardized per meter** using
`StandardScaler` (mean=0, std=1 per column). All loss values reported below
are computed **only on hole positions** (missing data), not on known values.

## Baselines

Baselines answer: _"How well can a simple, non-learned rule fill the holes?"_
Our neural network must **beat every baseline** to justify its existence.

| Baseline             | Strategy                                                 | Expected strength                           |
| -------------------- | -------------------------------------------------------- | ------------------------------------------- |
| **Zero (Mean)**      | Predict 0 everywhere (= the population mean)             | Weakest. RMSE ≈ 1.0 std.                    |
| **Forward Fill**     | Copy the last known value before the hole                | Decent for short gaps, fails for long ones. |
| **Linear Interp**    | Draw a straight line from last known to next known value | Strong for smooth data, fails on spikes.    |
| **Historical Daily** | Copy the value from exactly 24h ago (48 steps)           | Strong if consumption is diurnal.           |

### Measured Baseline Values (Numerical aproximations)

> `calculate_baselines(dm)`

| Baseline             | MSE        | RMSE (std) | Huber (δ=1.0) |
| -------------------- | ---------- | ---------- | ------------- |
| Zero (Mean)          | 0.9460     | **0.9726** | **0.3500**    |
| Forward Fill         | **1.5707** | **1.2533** | 0.4990        |
| Linear Interpolation | **1.2274** | **1.1079** | **0.4126**    |
| Historical Daily     | **1.0046** | **1.0023** | **0.3356**    |

The **strongest baseline** (lowest values) is our **minimum bar to beat**.

## Interpreting MSE on Standardized Data

Since each meter column is standardized (std = 1):

- **MSE** = average of (prediction − true)²
- **RMSE** = √MSE = average error **in standard deviations**

| MSE  | RMSE | Meaning                                          |
| ---- | ---- | ------------------------------------------------ |
| 1.00 | 1.00 | Predicting the mean. No information captured.    |
| 0.50 | 0.71 | ~71% of a std off. General direction only.       |
| 0.25 | 0.50 | ~50% of a std off. Shape visible, details wrong. |
| 0.09 | 0.30 | ~30% of a std off. Reasonable reconstruction.    |
| 0.04 | 0.20 | ~20% of a std off. Good quality.                 |
| 0.01 | 0.10 | ~10% of a std off. Excellent.                    |

### Converting to Original Units (Watts)

For any specific meter with original standard deviation σ_orig:

- Error in Watts = RMSE × σ_orig
- Example: if σ_orig = 500W and RMSE = 0.30 → average error ≈ 150W

To recover σ*orig per meter: `scaler_all.scale*[column_index]`

## Interpreting Huber Loss (δ=1.0) on Standardized Data

Huber Loss with δ=1.0:

- For |error| < 1.0 std: Huber = 0.5 × error² (same as 0.5 × MSE)
- For |error| ≥ 1.0 std: Huber = |error| − 0.5 (linear, caps the penalty)

Approximate conversion (depends on outlier frequency in data):

| Huber | ≈ MSE equivalent | ≈ RMSE | Quality                               |
| ----- | ---------------- | ------ | ------------------------------------- |
| 0.35  | ~1.0             | ~1.0   | Predicting the mean. Useless.         |
| 0.22  | ~0.50            | ~0.70  | General trend only. Not useful.       |
| 0.10  | ~0.22            | ~0.47  | Shape visible. Borderline.            |
| 0.05  | ~0.10            | ~0.32  | Reasonable. Acceptable for many uses. |
| 0.02  | ~0.04            | ~0.20  | Good reconstruction.                  |
| 0.01  | ~0.02            | ~0.14  | Excellent.                            |

## Target Values

### Minimum viable (must beat strongest baseline):

| Criterion | Target                     | Rationale                    |
| --------- | -------------------------- | ---------------------------- |
| MSE       | < strongest baseline MSE   | Must outperform simple rules |
| Huber     | < strongest baseline Huber | Must outperform simple rules |

### Visually acceptable (curve shape is recognizable):

| Criterion | Target                   |
| --------- | ------------------------ |
| MSE       | < 0.10 (RMSE < 0.32 std) |
| Huber     | < 0.05                   |

### Good quality (hard to distinguish from real at a glance):

| Criterion | Target                   |
| --------- | ------------------------ |
| MSE       | < 0.04 (RMSE < 0.20 std) |
| Huber     | < 0.02                   |

## Model Results Tracker

| Run | Architecture                       | Criterion | Window | Epochs | Val Loss | RMSE (approx) | vs Best Baseline |
| --- | ---------------------------------- | --------- | ------ | ------ | -------- | ------------- | ---------------- |
| v1  | BiLSTM(8L) + FFN + 128hd           | MSE       | 336    | 21     | 0.5859   | 0.77          | ???              |
| v2  | BiLSTM(4L) + Conv1D                | Huber     | 336    | ~12    | 0.648    | ~0.68         | ???              |
| v3  | BiLSTM(4L) + FFN + 128hd           | Huber     | 144    | 71     | 0.2196   | ~0.66         | ???              |
| v4  | BiLSTM(4L) + FFN + 128hd           | MSE       | 336    | 33     | 0.6022   | ~0.776        | ...              |
| v5  | BiLSTM(4L) + FFN (GELU) + 128hd    | Huber     | 336    | 39     | 0.2303   | ~0.678        | ...              |
| v6  | BiLSTM(4L) + Conv1D (GELU) + 128hd | Huber     | 336    | 30     | 0.2224   | ~0.667        | ...              |
| v7  | BiLSTM(16L) + FFN + 128hd          | MSE       | 336    | 11     | 0.9213   | ~0.96         | ...              |
| v8  | BiLSTM(4L) + LSTM + 128hd          | MSE       | 336    | 10     | 0.7368   | ~0.86         | ...              |

> Fill in "vs Best Baseline" column after running `calculate_baselines()`.
