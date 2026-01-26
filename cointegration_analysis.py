"""
GPU-Accelerated Cointegration Analysis
-------------------------------------

Purpose:
- Compute Engle–Granger residual-based ADF statistics ("scores") for all
    currency pairs over a rolling 7-day window, evaluating daily snapshots.

Tech:
- RAPIDS `cudf` for fast CSV I/O and column-wise joins on GPU
- `cupy` to vectorize OLS + residual ADF test for all pair combinations

Outputs:
- A list (per day) of pairwise results saved to results/daily_cointegration_results_gpu.pkl
- Each entry includes `pair1`, `pair2`, `score` (more negative ⇒ stronger),
    and a 5% significance flag (MacKinnon critical value).

Notes:
- This module focuses on clarity and performance without changing behavior.
- Prints give a concise progress readout; logging can be added if needed.
"""

import cudf
import cupy as cp
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from typing import Optional, Dict, Any
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/mnt/ssd2/DARWINEX_Mission/data")
OUTPUT_DIR = Path("/mnt/ssd2/DARWINEX_Mission/results")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "daily_cointegration_results_gpu.pkl"
WINDOW_DAYS = 7  # Rolling window length in days

# Critical values for Engle–Granger (2 variables, with constant)
# MacKinnon (1991, 2010) approximations
CRIT_1PCT = -3.90
CRIT_5PCT = -3.34
CRIT_10PCT = -3.04

def load_all_data_gpu(data_dir: Path) -> cudf.DataFrame:
    """
    Load all pair CSVs into a single wide `cudf.DataFrame`.

    - Reads only `timestamp` (ms since epoch) and `close`.
    - Renames `close` to the pair symbol (derived from filename).
    - Aligns by timestamp index, forward/backward fills small gaps.

    Returns a time-indexed DataFrame with one column per asset.
    """
    files = sorted(list(data_dir.glob("*-m1-*.csv")))
    if not files:
        raise ValueError(f"No data files found in {data_dir}")
    
    frames = []
    print(f"Reading {len(files)} files...")
    
    for f in tqdm(files, desc="Loading CSVs"):
        pair_name = f.name.split('-m1-')[0]
        # Read minimal columns for speed / memory; enforce dtypes
        df = cudf.read_csv(
            f,
            usecols=['timestamp', 'close'],
            dtype={'timestamp': 'int64', 'close': 'float32'}
        )
        df = df.rename(columns={'close': pair_name})
        df = df.set_index('timestamp')
        frames.append(df)
    
    print("Merging dataframes on GPU...")
    # Column-wise concat aligns on the index
    prices = cudf.concat(frames, axis=1)
    
    print("Sorting and filling NaNs...")
    prices = prices.sort_index()
    # Small gaps can occur due to market microstructure; fill conservatively
    prices = prices.ffill().bfill()
    
    # Convert index to datetime once
    prices.index = cudf.to_datetime(prices.index, unit='ms')
    
    return prices

def vectorized_eg_test(data_cp: cp.ndarray) -> cp.ndarray:
    """
    Compute Engle–Granger residual-based ADF t-statistics for all ordered pairs.

    Parameters
    ----------
    data_cp : cupy.ndarray
        Array of shape (T, N) with float32 prices (time × assets).

    Returns
    -------
    cupy.ndarray
        Matrix of shape (N, N) with t-stats for regression (y_i on x_j).
        Diagonal is NaN; lower/upper triangles contain directed tests.
    """
    T, N = data_cp.shape
    if T < 10:
        return cp.full((N, N), cp.nan)

    # 1) OLS components via moments: y_i = alpha_ij + beta_ij * x_j
    means = data_cp.mean(axis=0)
    data_centered = data_cp - means
    
    # Covariance matrix (N, N)
    cov = cp.dot(data_centered.T, data_centered) / (T - 1)
    var = cp.diag(cov)
    
    # Avoid division by zero
    var[var == 0] = 1e-10
    
    # beta_ij = cov_ij / var_j (for y_i = beta*x_j + alpha)
    beta = cov / var[None, :]
    
    # alpha_ij = mean_i - beta_ij * mean_j
    alpha = means[:, None] - beta * means[None, :]
    
    t_stats = cp.zeros((N, N), dtype=cp.float32)
    
    # 2) For each regressor x_j, compute residuals r_i = y_i - (alpha_ij + beta_ij x_j)
    #    and apply a no-constant ADF: Δr_i = γ_i r_{i,t-1} + ε_t
    for j in range(N):
        # Predicted y_i given regressor j, for all i at once → (T, N)
        y_pred = data_cp[:, [j]] * beta[:, j] + alpha[:, j]
        res = data_cp - y_pred 
        
        # ADF on residuals without constant term
        d_res = res[1:] - res[:-1]
        res_prev = res[:-1]
        
        # γ_i via OLS on Δr_i ~ r_{i,t-1}
        dot_num = cp.sum(d_res * res_prev, axis=0)
        dot_den = cp.sum(res_prev**2, axis=0)
        dot_den[dot_den == 0] = 1e-10
        
        gamma = dot_num / dot_den
        
        adf_res = d_res - res_prev * gamma
        rss = cp.sum(adf_res**2, axis=0)
        
        # Standard error of γ_i and t-statistic t_i = γ_i / se(γ_i)
        se = cp.sqrt((rss / (T - 2)) / dot_den)
        se[se == 0] = 1e-10
        
        t_stats[:, j] = gamma / se

    cp.fill_diagonal(t_stats, cp.nan)
    return t_stats

def analyze_window(
    df_window_gpu: cudf.DataFrame,
    window_start: datetime,
    window_end: datetime,
) -> Optional[Dict[str, Any]]:
    """
    Analyze all pairs for a single rolling window on GPU.

    Returns a dict with day metadata and sorted pair results, or None
    if the window is too short to be informative.
    """
    day_str = window_end.strftime("%Y-%m-%d")
    pairs = df_window_gpu.columns.tolist()
    
    # Use float32 for speed/memory on GPU
    data_cp = cp.asarray(df_window_gpu.values, dtype=cp.float32)
    T, N = data_cp.shape
    
    if T < 500:
        return None
        
    t_stats = vectorized_eg_test(data_cp)
    t_stats_cpu = cp.asnumpy(t_stats)
    
    results_list = []
    for i in range(N):
        for j in range(N):
            if i >= j: continue # Only upper triangle for unique pairs
            
            # Use the more significant direction between (i on j) and (j on i)
            score = min(t_stats_cpu[i, j], t_stats_cpu[j, i])
            
            if not np.isnan(score):
                results_list.append({
                    'pair1': pairs[i],
                    'pair2': pairs[j],
                    'score': float(score),
                    'is_significant_5pct': bool(score < CRIT_5PCT)
                })
    
    # Sort by score
    results_list.sort(key=lambda x: x['score'])
    
    min_s = results_list[0]['score'] if results_list else 0
    sig_count = sum(1 for x in results_list if x['is_significant_5pct'])
    print(
        f"📊 Day {day_str} (window {window_start.date()} → {window_end.date()}): "
        f"{len(results_list)} pairs. Significant(5%): {sig_count}. Min Score: {min_s:.4f}"
    )
    
    return {
        'day': day_str,
        'day_end': window_end,
        'window_start': window_start,
        'results': results_list,
        'pairs_count': N
    }

def main():
    """Entry point: orchestrates loading, rolling analysis, and persistence."""
    print("\n" + "="*70)
    print("🚀 ULTRA-FAST GPU COINTEGRATION ANALYSIS")
    print("="*70)
    
    start_time = datetime.now()
    
    df_all = load_all_data_gpu(DATA_DIR)
    
    import pandas as pd
    start_date = pd.Timestamp(df_all.index.min()).floor('D')
    end_date = pd.Timestamp(df_all.index.max())
    
    print(f"\nData range: {start_date} to {end_date}")
    
    # First evaluable day is the end of the initial 7-day window
    current_day = start_date + timedelta(days=WINDOW_DAYS - 1)
    days_data = []
    
    while current_day <= end_date:
        window_start = current_day - timedelta(days=WINDOW_DAYS - 1)
        df_window = df_all.loc[window_start:current_day]
        
        if len(df_window) > 500:
            res = analyze_window(df_window, window_start, current_day)
            if res:
                days_data.append(res)
        
        current_day = current_day + timedelta(days=1)
    
    # Save results
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(days_data, f)
    
    duration = datetime.now() - start_time
    print(f"\n✅ Done! Processed {len(days_data)} rolling days in {duration.total_seconds():.1f}s.")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()