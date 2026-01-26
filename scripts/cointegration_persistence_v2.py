"""
GPU-Accelerated Cointegration Persistence V2: Fixed Beta Analysis
-----------------------------------------------------------------
1. Finds cointegrated pairs at Day T (Optimized Beta).
2. Tests those SPECIFIC pairs at Day T+1 using the FROZEN Beta from Day T.
   (Does the old hedge ratio still produce a stationary spread the next day?)
3. Compares Windows: 14, 30, 90 days.
"""

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
RESULTS_FILE = OUTPUT_DIR / "persistence_v2_fixed_beta.pkl"
WINDOW_SIZES = [14, 30, 90]
CRIT_5PCT = -3.34

def load_all_data_gpu(data_dir: Path) -> cudf.DataFrame:
    files = sorted(list(data_dir.glob("DAT_ASCII_*.csv")))
    if not files: raise ValueError("No data files")
    frames = []
    print(f"Reading {len(files)} files...")
    for f in tqdm(files):
        try:
            pair_name = f.name.split('_')[2].lower()
            df = cudf.read_csv(f, sep=';', header=None, usecols=[0, 4], names=['dt', 'close'], dtype={'dt':'str', 'close':'float32'})
            df['timestamp'] = cudf.to_datetime(df['dt'], format='%Y%m%d %H%M%S')
            df = df.drop_duplicates('timestamp').drop(columns='dt').rename(columns={'close': pair_name}).set_index('timestamp')
            frames.append(df)
        except: continue
    return cudf.concat(frames, axis=1).sort_index().ffill().bfill()

def adf_no_constant_score(residuals: cp.ndarray):
    """Computes ADF t-stat on residuals without constant (approx)."""
    # d_res = gamma * res_prev + err
    d_res = residuals[1:] - residuals[:-1]
    res_prev = residuals[:-1]
    
    # OLS: gamma = (y . x) / (x . x)
    num = cp.sum(d_res * res_prev)
    den = cp.sum(res_prev**2)
    if den == 0: return 0.0
    
    gamma = num / den
    err = d_res - gamma * res_prev
    rss = cp.sum(err**2)
    
    n = len(d_res)
    if n <= 2: return 0.0
    
    se = cp.sqrt(rss / (n - 1)) / cp.sqrt(den)
    if se == 0: return 0.0
    
    return float(gamma / se)

def analyze_window_optimized(data_cp: cp.ndarray, pairs: list):
    """
    Standard Engle-Granger: Optimizes Beta for current window.
    Returns list of dicts for significant pairs.
    """
    T, N = data_cp.shape
    if T < 10: return []

    means = data_cp.mean(axis=0)
    centered = data_cp - means
    cov = cp.dot(centered.T, centered) / (T - 1)
    var = cp.diag(cov)
    var[var == 0] = 1e-10
    
    beta = cov / var[None, :] # beta_ij = cov_ij / var_j
    alpha = means[:, None] - beta * means[None, :]
    
    results = []
    
    # We loop to run ADFs. Vectorizing fully is complex, loop is okay for N=27
    for j in range(N): # Regressor (X)
        y_preds = data_cp[:, [j]] * beta[:, j] + alpha[:, j] # (T, N)
        residuals_all = data_cp - y_preds
        
        # Calculate ADF for each column i (Dependent Y)
        # Vectorized ADF-ish loop
        for i in range(N):
            if i == j: continue
            
            # Extract single series for ADF
            res = residuals_all[:, i]
            score = adf_no_constant_score(res)
            
            if score < CRIT_5PCT:
                results.append({
                    'y': pairs[i],
                    'x': pairs[j],
                    'beta': float(beta[i, j]),
                    'alpha': float(alpha[i, j]),
                    'score': score
                })
    return results

def check_fixed_beta_next_day(data_next_cp: cp.ndarray, candidates: list, pair_map: dict):
    """
    Checks stored candidates on the NEW window (shifted by 1 day)
    using the OLD beta/alpha.
    """
    if data_next_cp.shape[0] < 10: return []
    
    verified = []
    for cand in candidates:
        idx_y = pair_map[cand['y']]
        idx_x = pair_map[cand['x']]
        
        # Reconstruct spread on NEW data using OLD params
        # Spread = Y - (Beta*X + Alpha)
        Y = data_next_cp[:, idx_y]
        X = data_next_cp[:, idx_x]
        
        # We test stationarity of the spread
        # Note: Ideally we re-center (remove mean) because 'Alpha' absorbs the mean level.
        # If the spread drifts, ADF without constant might fail. 
        # Standard EG checks residuals. Spread = Y - bX.
        # Let's use Spread - mean(Spread) to be fair to the "Stationarity" definition.
        
        spread = Y - (cand['beta'] * X + cand['alpha'])
        spread_centered = spread - cp.mean(spread)
        
        score_fixed = adf_no_constant_score(spread_centered)
        
        cand['score_next_day'] = score_fixed
        verified.append(cand)
        
    return verified

def main():
    print("🚀 Starting FIXED BETA Persistence Analysis...")
    df_all = load_all_data_gpu(DATA_DIR)
    pairs = df_all.columns.tolist()
    pair_map = {p: i for i, p in enumerate(pairs)}
    
    import pandas as pd
    start_date = pd.Timestamp(df_all.index.min()).floor('D')
    end_date = pd.Timestamp(df_all.index.max())
    
    final_results = {}
    
    for w_size in WINDOW_SIZES:
        print(f"\n👉 Window: {w_size}d")
        
        # We need to access data by integer index for speed
        # Convert entire dataframe to cupy once? No, it's large.
        # But 1 min data for 1 year is ~370k rows * 27 cols * 4 bytes ~= 40MB.
        # Fits easily in GPU memory.
        data_matrix_all = cp.asarray(df_all.values, dtype=cp.float32)
        timestamps = df_all.index.to_pandas()
        
        # Map dates to indices roughly or iterate by index
        # 1440 minutes per day. 
        # Better: iterate by calendar day, lookup index range.
        
        # Pre-calculate day indices
        # This acts as a lookup table: Date -> (start_idx, end_idx)
        # Actually, rolling window by TIME is safer.
        
        day_indices = []
        curr = start_date
        while curr <= end_date:
            # Find indices for this day
            mask = (timestamps >= curr) & (timestamps < curr + timedelta(days=1))
            if mask.any():
                day_indices.append(curr)
            curr += timedelta(days=1)
            
        window_stats = []
        
        # We need at least w_size days of history
        # Start loop from w_size-th available day
        for i in tqdm(range(w_size, len(day_indices) - 1)):
            day_t = day_indices[i]
            day_t_next = day_indices[i+1] # We check t+1
            
            # Window T: [t - w_size + 1, t]
            # End of day t
            end_ts_t = day_t + timedelta(days=1)
            start_ts_t = end_ts_t - timedelta(days=w_size)
            
            mask_t = (timestamps >= start_ts_t) & (timestamps < end_ts_t)
            data_t = data_matrix_all[mask_t]
            
            # Window T+1: [t - w_size + 2, t+1] (Shifted 1 day forward)
            end_ts_next = day_t_next + timedelta(days=1)
            start_ts_next = end_ts_next - timedelta(days=w_size)
            
            mask_next = (timestamps >= start_ts_next) & (timestamps < end_ts_next)
            data_next = data_matrix_all[mask_next]
            
            if len(data_t) < 100 or len(data_next) < 100:
                continue
                
            # 1. Optimize on T
            candidates_t = analyze_window_optimized(data_t, pairs)
            
            if not candidates_t:
                continue
                
            # 2. Check on T+1 (Fixed Beta)
            results_t1 = check_fixed_beta_next_day(data_next, candidates_t, pair_map)
            
            window_stats.append({
                'date': day_t.strftime('%Y-%m-%d'),
                'data': results_t1
            })
            
        final_results[w_size] = window_stats
        
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(final_results, f)
    print(f"\n✅ Done. Saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
