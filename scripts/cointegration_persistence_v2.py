"""
GPU-Accelerated Cointegration Persistence V2: Fixed Beta Analysis (Multi-Stream Optimized)
------------------------------------------------------------------------------------------
1. Single pass over chronological days.
2. Uses CUDA Streams to compute windows [14, 30, 90] in PARALLEL on the GPU for each day.
3. Minimizes CPU-GPU synchronization points.
4. Uses TLS (Total Least Squares) instead of OLS for cointegration estimation.

Performance:
- Reduced index lookups (3x -> 1x).
- Overlapped Kernel Execution via Streams.
"""

import cudf
import cupy as cp
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
import pandas as pd

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

def get_tls_params(data_cp: cp.ndarray):
    """
    Compute TLS (Total Least Squares) parameters via vectorized eigenvalue decomposition.
    
    For each regression (y_i on x_j), computes the minimum eigenvector of the 2x2 
    covariance matrix [[var_y, cov_yx], [cov_yx, var_x]] to find the best-fit line
    that minimizes orthogonal distance (not OLS residuals).
    
    FULLY VECTORIZED: No explicit loops. All pairs computed simultaneously on GPU.
    
    Returns:
    - beta: (N, N) matrix of TLS slopes
    - alpha: (N, N) matrix of TLS intercepts
    """
    T, N = data_cp.shape
    
    means = data_cp.mean(axis=0)
    centered = data_cp - means
    
    # Build full covariance matrix (N, N)
    cov = cp.dot(centered.T, centered) / (T - 1)
    diag_cov = cp.diag(cov)  # (N,) - diagonal variance vector
    
    # Broadcasting: For pair (i, j):
    #   a[i, j] = var(y_i) = cov[i, i]
    #   b[i, j] = var(x_j) = cov[j, j]
    #   c[i, j] = cov(y_i, x_j) = cov[i, j]
    a = diag_cov[:, None]  # (N, 1) broadcasts to (N, N)
    b = diag_cov[None, :]  # (1, N) broadcasts to (N, N)
    c = cov  # (N, N)
    
    # For 2x2 matrix [[a, c], [c, b]], compute minimum eigenvalue using closed form:
    # trace = a + b
    # det = a*b - c^2
    # lambda_min = (trace - sqrt(trace^2 - 4*det)) / 2
    #           = (a + b)/2 - sqrt(((a-b)/2)^2 + c^2)
    
    trace = a + b
    srt = cp.sqrt(((a - b) / 2)**2 + c * c)
    lambda_min = trace / 2 - srt
    
    # Eigenvector for lambda_min: [c, lambda_min - a]
    v0 = c
    v1 = lambda_min - a
    
    # TLS slope: beta = -v0 / v1 (where v = [v0, v1])
    # Avoid division by zero
    beta = cp.where(cp.abs(v1) > 1e-10, -v0 / v1, 0.0)
    
    # Set diagonal to 1.0 (identity regression)
    cp.fill_diagonal(beta, 1.0)
    
    # Intercept: alpha = mean_y - beta * mean_x
    alpha = means[:, None] - beta * means[None, :]
    
    return beta, alpha

def compute_adf_stats(data_cp: cp.ndarray, beta: cp.ndarray, alpha: cp.ndarray):
    """Computes ADF t-statistics for spread."""
    T, N = data_cp.shape
    t_stats = cp.zeros((N, N), dtype=cp.float32)
    
    for j in range(N):
        x_j = data_cp[:, j]
        y_preds = x_j[:, None] * beta[:, j][None, :] + alpha[:, j][None, :]
        spread = data_cp - y_preds
        
        spread_mean = spread.mean(axis=0)
        spread_centered = spread - spread_mean
        
        res = spread_centered
        d_res = res[1:] - res[:-1]
        res_prev = res[:-1]
        
        dot_num = cp.sum(d_res * res_prev, axis=0)
        dot_den = cp.sum(res_prev**2, axis=0)
        dot_den[dot_den == 0] = 1e-10
        
        gamma = dot_num / dot_den
        
        adf_res = d_res - res_prev * gamma[None, :]
        rss = cp.sum(adf_res**2, axis=0)
        
        se = cp.sqrt((rss / (T - 2)) / dot_den)
        se[se == 0] = 1e-10
        
        t_stats[:, j] = gamma / se
        
    return t_stats

def main():
    print("🚀 Starting FIXED BETA Persistence Analysis (Multi-Stream Optimized)...")
    
    start_time_total = datetime.now()
    df_all = load_all_data_gpu(DATA_DIR)
    pairs = df_all.columns.tolist()
    
    start_date = pd.Timestamp(df_all.index.min()).floor('D')
    end_date = pd.Timestamp(df_all.index.max())
    
    print("Converting data to GPU matrix...")
    data_matrix_all = cp.asarray(df_all.values, dtype=cp.float32)
    timestamps = df_all.index.to_pandas()
    
    print("Indexing dates...")
    ts_values = pd.to_datetime(timestamps).values # Ensure numpy array of datetime64
    unique_dates = pd.to_datetime(timestamps).floor('D').unique().sort_values()
    date_to_idx = {}
    
    for d in unique_dates:
        ts_start = d.to_datetime64()
        ts_end = (d + timedelta(days=1)).to_datetime64()
        idx_start = np.searchsorted(ts_values, ts_start, side='left')
        idx_end = np.searchsorted(ts_values, ts_end, side='left')
        
        if idx_end > idx_start:
            date_to_idx[d] = (idx_start, idx_end)
            
    sorted_days = sorted(list(date_to_idx.keys()))
    print(f"Indexed {len(sorted_days)} days.")

    # Initialize results containers
    final_results = {w: [] for w in WINDOW_SIZES}
    
    # Create streams for each window size to run in parallel
    streams = {w: cp.cuda.Stream() for w in WINDOW_SIZES}
    
    # Pre-allocate containers for GPU results to keep them alive until sync
    # Dictionary mapping window_size -> dict of results for current day
    
    print("Processing days...")
    for i in tqdm(range(len(sorted_days) - 1)):
        day_t = sorted_days[i]
        day_t1 = sorted_days[i+1]
        
        if (day_t1 - day_t).days > 5: continue
        
        # CPU Indices lookup
        idx_day_t_end = date_to_idx[day_t][1]
        idx_day_t1_end = date_to_idx[day_t1][1]
        
        # Holder for current day's GPU arrays
        gpu_results = {} 
        
        # --- LAUNCH KERNELS (Parallel Dispatch) ---
        for w_size in WINDOW_SIZES:
            with streams[w_size]:
                # 1. Slice Windows (Indices calc on CPU, Slicing on GPU view)
                ts_win_start = (day_t + timedelta(days=1) - timedelta(days=w_size)).to_datetime64()
                idx_win_start = np.searchsorted(ts_values, ts_win_start, side='left')
                
                ts_win_t1_start = (day_t1 + timedelta(days=1) - timedelta(days=w_size)).to_datetime64()
                idx_win_t1_start = np.searchsorted(ts_values, ts_win_t1_start, side='left')
                
                # Validation
                if idx_day_t_end <= idx_win_start or idx_day_t1_end <= idx_win_t1_start:
                    gpu_results[w_size] = None
                    continue
                
                data_t = data_matrix_all[idx_win_start : idx_day_t_end]
                data_t1 = data_matrix_all[idx_win_t1_start : idx_day_t1_end]
                
                if len(data_t) < 100 or len(data_t1) < 100:
                    gpu_results[w_size] = None
                    continue
                
                # 2. Train on T
                beta, alpha = get_tls_params(data_t)
                
                if beta is None:
                    gpu_results[w_size] = None
                    continue
                
                scores_t = compute_adf_stats(data_t, beta, alpha)
                
                # 3. Test on T+1 (Fixed Beta)
                scores_t1 = compute_adf_stats(data_t1, beta, alpha)

                # 4. Train on T+1 (Optimal Beta)
                beta_t1, alpha_t1 = get_tls_params(data_t1)
                
                # Store GPU arrays
                gpu_results[w_size] = (scores_t, scores_t1, beta, beta_t1)

        # --- SYNCHRONIZE & COPY BACK (Sequential) ---
        # We assume all streams launched. Now we gather.
        # cp.asnumpy will synchronize the specific stream needed for that array.
        
        for w_size in WINDOW_SIZES:
            res = gpu_results.get(w_size)
            if res is None: continue
            
            scores_t_gpu, scores_t1_gpu, beta_gpu, beta_t1_gpu = res
            
            # Explicit sync not strictly needed if using same stream context, 
            # but asnumpy handles it.
            # We are in default stream here, accessing arrays from other streams.
            # CuPy handles dependency if we fetch them.
            
            h_scores_t = cp.asnumpy(scores_t_gpu)
            h_scores_t1 = cp.asnumpy(scores_t1_gpu)
            h_beta = cp.asnumpy(beta_gpu)
            h_beta_t1 = cp.asnumpy(beta_t1_gpu)
            
            # --- CPU Filtering ---
            sig_indices = np.argwhere(h_scores_t < CRIT_5PCT)
            day_list = []
            for idx in sig_indices:
                r, c = idx
                if r == c: continue
                day_list.append({
                    'y': pairs[r],
                    'x': pairs[c],
                    'beta': float(h_beta[r, c]),
                    'beta_next': float(h_beta_t1[r, c]),
                    'score': float(h_scores_t[r, c]),
                    'score_next_day': float(h_scores_t1[r, c])
                })
            
            if day_list:
                final_results[w_size].append({
                    'date': day_t.strftime('%Y-%m-%d'),
                    'data': day_list
                })

    # Save
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(final_results, f)
        
    print(f"\n✅ Done in {(datetime.now() - start_time_total).total_seconds():.1f}s. Saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
