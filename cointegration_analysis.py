"""
Cointegration Analysis Module - Ultra-Fast GPU Optimized
Analyzes cointegration of asset pairs on a rolling daily basis (last 7 days).
Uses RAPIDS (cudf) for fast I/O and CuPy for vectorized statistical tests on GPU.
"""

import os
import gc
import cudf
import cupy as cp
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/mnt/ssd2/DARWINEX_Mission/data")
OUTPUT_DIR = Path("/mnt/ssd2/DARWINEX_Mission/results")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "daily_cointegration_results_gpu.pkl"
WINDOW_DAYS = 7

# Critical values for Engle-Granger (2 variables, with constant)
# From MacKinnon (1991, 2010)
CRIT_1PCT = -3.90
CRIT_5PCT = -3.34
CRIT_10PCT = -3.04

def load_all_data_gpu(data_dir: Path) -> cudf.DataFrame:
    """Load all pair CSVs into a single wide cudf DataFrame efficiently"""
    files = sorted(list(data_dir.glob("*-m1-*.csv")))
    if not files:
        raise ValueError(f"No data files found in {data_dir}")
    
    all_dfs = []
    print(f"Reading {len(files)} files...")
    
    for f in tqdm(files, desc="Loading CSVs"):
        pair_name = f.name.split('-m1-')[0]
        # Read only timestamp and close
        df = cudf.read_csv(f, usecols=['timestamp', 'close'], dtype={'timestamp': 'int64', 'close': 'float32'})
        df = df.rename(columns={'close': pair_name})
        df = df.set_index('timestamp')
        all_dfs.append(df)
    
    print("Merging dataframes on GPU...")
    # Use axis=1 concat for alignment
    final_df = cudf.concat(all_dfs, axis=1)
    
    print("Sorting and filling NaNs...")
    final_df = final_df.sort_index()
    # Forward fill then backward fill
    final_df = final_df.ffill().bfill()
    
    # Convert index to datetime once
    final_df.index = cudf.to_datetime(final_df.index, unit='ms')
    
    return final_df

def vectorized_eg_test(data_cp: cp.ndarray):
    """
    Perform Engle-Granger Cointegration Test for all pairs (i, j) on GPU.
    data_cp: (T, N) array of asset prices
    Returns: t_stats matrix (N, N)
    """
    T, N = data_cp.shape
    if T < 10:
        return cp.full((N, N), cp.nan)

    # 1. Compute means and variances/covariances for OLS: y = beta*x + alpha
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
    
    # Vectorize over i (dependent), loop over j (independent)
    for j in range(N):
        # x_j is data_cp[:, j]
        # y_pred for all i: (T, N)
        y_pred = data_cp[:, [j]] * beta[:, j] + alpha[:, j]
        res = data_cp - y_pred 
        
        # ADF Test on residuals (no constant)
        d_res = res[1:] - res[:-1]
        res_prev = res[:-1]
        
        # gamma = sum(d_res * res_prev) / sum(res_prev^2)
        dot_num = cp.sum(d_res * res_prev, axis=0)
        dot_den = cp.sum(res_prev**2, axis=0)
        dot_den[dot_den == 0] = 1e-10
        
        gamma = dot_num / dot_den
        
        adf_res = d_res - res_prev * gamma
        rss = cp.sum(adf_res**2, axis=0)
        
        # standard error of gamma
        se = cp.sqrt((rss / (T - 2)) / dot_den)
        se[se == 0] = 1e-10
        
        t_stats[:, j] = gamma / se

    cp.fill_diagonal(t_stats, cp.nan)
    return t_stats

def analyze_window(df_window_gpu: cudf.DataFrame, window_start, window_end):
    """Analyze all pairs for a given rolling window fully on GPU"""
    day_key = window_end.strftime("%Y-%m-%d")
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
            
            # For each pair (i, j), we actually have two possible regressions:
            # y=i, x=j and y=j, x=i. coint() typically tests one.
            # We'll take the minimum (most significant) of the two.
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
    print(f"📊 Day {day_key} (window {window_start.date()} → {window_end.date()}): {len(results_list)} pairs. Significant(5%): {sig_count}. Min Score: {min_s:.4f}")
    
    return {
        'day': day_key,
        'day_end': window_end,
        'window_start': window_start,
        'results': results_list,
        'pairs_count': N
    }

def main():
    print("\n" + "="*70)
    print("🚀 ULTRA-FAST GPU COINTEGRATION ANALYSIS")
    print("="*70)
    
    start_time = datetime.now()
    
    df_all = load_all_data_gpu(DATA_DIR)
    
    import pandas as pd
    start_date = pd.Timestamp(df_all.index.min()).floor('D')
    end_date = pd.Timestamp(df_all.index.max())
    
    print(f"\nData range: {start_date} to {end_date}")
    
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