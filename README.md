# GPU-Accelerated Rolling Cointegration Analysis (FX)

A high-performance pipeline to compute Engle–Granger cointegration statistics across currency pairs using RAPIDS `cudf` for GPU I/O and `cupy` for vectorized computation. Results are analyzed in a notebook with clear, multi-view visualizations and persistence metrics.

---

## Overview

- **Goal:** Identify and track short-horizon cointegration relationships among FX pairs using a 7-day rolling window and daily evaluations.
- **Core Idea:** For each ordered pair (asset `i` as dependent, asset `j` as regressor), run OLS, compute residuals, then perform a residual-based ADF test. The t-statistic is the "score"; more negative values indicate stronger cointegration.
- **GPU:** All heavy operations (CSV reading, alignment, OLS moments, ADF computation) are performed on the GPU for speed.
- **Outputs:** Daily results are saved to `results/daily_cointegration_results_gpu.pkl` and visualized in the notebook `cointegration_analysis.ipynb`.

---

## Data & Preprocessing

- **Input format:** CSVs named like `eurusd-m1-YYYY-MM-DD-to-YYYY-MM-DD.csv` with columns: `timestamp` (ms since epoch) and `close`.
- **Alignment:** Files are merged on `timestamp` (index). We apply forward-fill then backward-fill to handle microstructure gaps conservatively.
- **Dtypes:** Prices are cast to `float32` to reduce GPU memory footprint.

---

## Methodology

### 1) OLS via Moments
For prices matrix $X \in \mathbb{R}^{T\times N}$ (time $T$, assets $N$), the regression for ordered pair $(i, j)$ is:

$$ y_{t}^{(i)} = \alpha_{ij} + \beta_{ij} x_{t}^{(j)} + \varepsilon_t $$

We estimate using moments:
- Centered data: $\tilde{X} = X - \mathrm{mean}(X)$
- Covariance: $\mathrm{Cov} = \frac{\tilde{X}^\top \tilde{X}}{T - 1}$
- Variance of regressor $j$: $\sigma_j^2 = \mathrm{Cov}_{jj}$
- Slope: $\beta_{ij} = \frac{\mathrm{Cov}_{ij}}{\sigma_j^2}$
- Intercept: $\alpha_{ij} = \mathrm{mean}(y^{(i)}) - \beta_{ij}\, \mathrm{mean}(x^{(j)})$

This formulation avoids per-pair OLS loops and leverages GPU matrix ops.

### 2) Engle–Granger Residual-Based ADF
Residuals for $(i, j)$:

$$ r_t^{(i|j)} = y_t^{(i)} - (\alpha_{ij} + \beta_{ij} \, x_t^{(j)}) $$

We apply an ADF without constant:

$$ \Delta r_t = \gamma \, r_{t-1} + \epsilon_t $$

Estimated via OLS:

$$ \hat{\gamma} = \frac{\sum_t (\Delta r_t \cdot r_{t-1})}{\sum_t r_{t-1}^2} $$

Residuals of this regression ($\Delta r_t - \hat{\gamma} r_{t-1}$) yield RSS. The standard error follows:

$$ \mathrm{se}(\hat{\gamma}) = \sqrt{ \frac{\mathrm{RSS}/(T - 2)}{\sum_t r_{t-1}^2} } $$

The **t-statistic (score)** is:

$$ t = \frac{\hat{\gamma}}{\mathrm{se}(\hat{\gamma})} $$

- **Interpretation:** More negative $t$ implies stronger evidence of cointegration (the residual series is more stationary).
- **Direction choice:** For each unordered pair \{i, j\}, we compute both directions $(i\leftarrow j)$ and $(j\leftarrow i)$ and take the minimum (most negative) as the pair score.

### 3) Statistical Thresholds
MacKinnon critical values for 2-variable EG test (with constant):

- 1%: $-3.90$
- 5%: $-3.34$ (primary threshold used)
- 10%: $-3.04$

A pair is considered **significant** at 5% if $t < -3.34$.

### 4) Rolling Window & Daily Evaluation
- **Window:** `WINDOW_DAYS = 7`. Each daily snapshot evaluates the preceding 7 calendar days.
- **Minimum samples:** Windows with fewer than ~500 observations are skipped to avoid noisy estimates.
- **Result structure:** For each day:
  - `day` (YYYY-MM-DD), `day_end`, `window_start`
  - `results`: list of `{pair1, pair2, score, is_significant_5pct}` sorted by `score` ascending
  - `pairs_count`: number of assets analyzed

---

## Outputs

- **File:** `results/daily_cointegration_results_gpu.pkl`
- **Type:** Python `list` of per-day dicts (see structure above)
- **Usage:** Loaded by the notebook for matrix construction and visualization.

---

## Notebook: Analytic Views & Metrics

The notebook `cointegration_analysis.ipynb` transforms the saved results into multiple complementary views:

### A) Matrices for Heatmaps
- Build a symmetric score matrix per day where entry $(i, j)$ is the pair score.
- Assets are sorted and mapped to indices for consistent axes.
- Used to display **cointegration intensity** across all pairs simultaneously.

### B) Cointegration Score Heatmaps
- **Colormap:** `RdYlGn_r` (darker ⇒ more negative ⇒ stronger).
- **Center:** Visual emphasis at the 5% threshold ($-3.34$).
- **Interpretation:** Blocks of persistent dark regions suggest stable cointegration clusters.

### C) Top Pairs Over Time (Multi-Panel)
From the top-5 pairs per day, we track the most frequent 10:
- **Panel 1:** Lines + markers (time vs score) to show trajectory.
- **Panel 2:** Pure scatter for reduced visual bias.
- **Panel 3:** Box plots by pair to summarize dispersion and stability.
- **Panel 4:** Mean score per pair with standard deviation as error bars; a vertical line at the 5% threshold; axis reversed so stronger is visually rightward.

### D) Top 20 Consistent Pairs & Persistence
- **Selection:** Top-20 pairs per day; count frequency across days.
- **Ordering:** Pairs ordered by mean score (ascending, stronger first).
- **Metrics:**
  - **Survival %:** For each day $t$, does a top-20 pair remain significant at day $t+1$? Percentage aggregated by pair.
  - **Mean Score:** Average score across appearances (colorscale conveys strength).

### E) Persistence Analysis (No Lookahead Bias)
- For each day $t$, pick top-20 pairs and compare $\text{Score}_t$ vs $\text{Score}_{t+1}$.
- **Scatter:** Shows transitions; a dashed diagonal indicates perfect persistence.
- **Binned Survival:** 7-day resampled survival rates give smoothed stability measure.

### F) Animated Cointegration Matrix
- Frame-by-frame heatmaps over days show evolution of the full matrix.
- Useful to observe **emergent clusters** or **regime shifts**.

---

## Performance Considerations

- **Vectorization:** OLS and ADF are computed in batch form across assets/pairs.
- **Precision:** `float32` balances speed and memory; thresholds are robust to this precision.
- **Complexity:** Approximately $O(N^2 T)$ but optimized via GPU throughput.
- **Gaps:** Conservative ffill/bfill minimizes distortions from sparse timestamps.

---

## Reproducibility & Running

### Environment
- Tested with Conda environment `rapids-23.12`.
- GPU support for `cudf` and `cupy` required.

### Run the Analysis Script
```bash
conda activate rapids-23.12
python /mnt/ssd2/DARWINEX_Mission/cointegration_analysis.py
```

### Visualize in the Notebook
- Open `cointegration_analysis.ipynb` and run the cells.
- The notebook reads `results/daily_cointegration_results_gpu.pkl` and renders the views.

---

## API Snapshot (Saved Results)
Each daily entry:
- `day`: string (YYYY-MM-DD)
- `day_end`: pandas `Timestamp`
- `window_start`: pandas `Timestamp`
- `pairs_count`: int
- `results`: list of dicts
  - `pair1`, `pair2`: symbols (e.g., `eurusd`, `usdjpy`)
  - `score`: float (t-stat; more negative ⇒ stronger)
  - `is_significant_5pct`: bool (`score < -3.34`)

---

## Extensions & Options

- **Logging:** Replace prints with `logging` and configurable verbosity.
- **CPU Fallback:** Use `pandas` + `numpy` when GPU isn’t available.
- **Parameterization:** CLI args for `WINDOW_DAYS`, input/output paths, sample thresholds.
- **Robustness:** Add outlier filtering or microstructure-aware resampling.

---

## Glossary

- **Engle–Granger Test:** A two-step approach: OLS to form residuals, then stationarity test (ADF) on residuals. Evidence of stationarity suggests cointegration.
- **ADF (Augmented Dickey–Fuller):** Tests if a time series has a unit root; here applied to residuals.
- **MacKinnon Critical Values:** Empirical critical values used to assess significance of unit root tests.
- **Cointegration:** Linear combination of non-stationary series is stationary; implies long-run equilibrium relationship.

---

## Files

- `cointegration_analysis.py` — GPU pipeline to compute daily rolling EG scores and save results.
- `cointegration_analysis.ipynb` — Visual analytics, heatmaps, persistence, and animation.
- `data/` — Input CSVs (timestamp ms + close).
- `results/` — Saved pickle with daily results.
