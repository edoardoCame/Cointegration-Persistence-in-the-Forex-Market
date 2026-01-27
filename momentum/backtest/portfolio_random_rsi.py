
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from numba import cuda
import math

# Set up styles
plt.style.use('bmh')
sns.set_context("talk")

# ------------------------------------------------------------------------------
# CUDA KERNELS
# ------------------------------------------------------------------------------

@cuda.jit
def rsi_kernel_cuda(gains, losses, lookbacks, out_rsi):
    lb_idx = cuda.grid(1)
    if lb_idx >= lookbacks.shape[0]:
        return
        
    n = lookbacks[lb_idx]
    
    # alpha = 1/n for Wilder's equivalent in pandas ewm(alpha=1/n, adjust=False)
    alpha = 1.0 / n
    run_alpha = 1.0 - alpha
    
    n_samples = gains.shape[0]
    
    # Initialize with first value
    avg_gain = gains[0]
    avg_loss = losses[0]
    
    # First point
    if avg_loss == 0:
        out_rsi[lb_idx, 0] = 100.0
    else:
        rs = avg_gain / avg_loss
        out_rsi[lb_idx, 0] = 100.0 - (100.0 / (1.0 + rs))

    # Loop
    for i in range(1, n_samples):
        g = gains[i]
        l = losses[i]
        
        # update
        avg_gain = run_alpha * avg_gain + alpha * g
        avg_loss = run_alpha * avg_loss + alpha * l
        
        # Masks for min_periods match
        # Pandas ewm min_periods=lb means first lb-1 are NaN.
        # But we accept warmup.
        if i < n - 1:
            out_rsi[lb_idx, i] = np.nan
        else:
            if avg_loss == 0:
                out_rsi[lb_idx, i] = 100.0
            else:
                rs = avg_gain / avg_loss
                out_rsi[lb_idx, i] = 100.0 - (100.0 / (1.0 + rs))

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def load_data_cudf(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "DAT_ASCII_GBPJPY_M1_20*.csv")))
    print(f"Found {len(files)} files.")
    dfs = []
    for f in files:
        # Format: 20230101 170400;1.069700;1.069740;1.069700;1.069700;0
        try:
            df = cudf.read_csv(f, sep=';', names=['date_str', 'open', 'high', 'low', 'close', 'volume'], header=None)
            df['time'] = cudf.to_datetime(df['date_str'].astype(str), format='%Y%m%d %H%M%S')
            df = df.drop(columns=['date_str', 'open', 'high', 'low', 'volume']) # optimization: keep only close/time
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        raise ValueError("No data loaded")
        
    full_df = cudf.concat(dfs).sort_values('time').reset_index(drop=True)
    return full_df

def calculate_rsi_matrix_gpu(close_array_gpu, lookbacks):
    """
    Calculates RSI for multiple lookbacks using Numba CUDA Kernel.
    Returns: (N_LB, Time) matrix (CuPy array)
    """
    # Pre-calc gains/losses on GPU (vectorized)
    # close_array_gpu is cupy array
    delta = cp.diff(close_array_gpu)
    # diff length is N-1. Prepend 0.0 to match length N.
    delta_padded = cp.concatenate((cp.array([0.0], dtype=cp.float32), delta))
    
    gain = delta_padded.copy()
    loss = delta_padded.copy()
    gain[gain < 0] = 0.0
    loss[loss > 0] = 0.0
    loss = -loss
    
    # Inputs for kernel
    lookbacks_gpu = cp.array(lookbacks, dtype=cp.int32)
    
    n_lb = len(lookbacks)
    n_time = len(close_array_gpu)
    
    out_rsi = cp.full((n_lb, n_time), np.nan, dtype=cp.float32)
    
    # Kernel Launch
    threads_per_block = 128
    blocks = (n_lb + threads_per_block - 1) // threads_per_block
    
    print(f"Calculating RSI on GPU for {n_lb} lookbacks...")
    t0 = time.time()
    
    rsi_kernel_cuda[blocks, threads_per_block](gain, loss, lookbacks_gpu, out_rsi)
    cuda.synchronize()
    
    print(f"RSI GPU Calc finished in {time.time()-t0:.4f}s")
    
    return out_rsi

def run_portfolio():
    # --------------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------------
    # Range: 60 to 30000
    POSSIBLE_LOOKBACKS = np.arange(60, 30001, 60)
    # Thresholds
    POSSIBLE_UPPER = np.arange(55, 96, 5) 
    POSSIBLE_LOWER = np.arange(5, 46, 5)
    
    N_STRATEGIES = 10
    COST_PER_FLIP = 0.012 # 1.2 pips (JPY pairs)
    INITIAL_CAPITAL = 10000.0
    
    # PATHS
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_portfolio")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. SELECT RANDOM STRATEGIES
    # --------------------------------------------------------------------------
    print("Selecting 10 Random Strategies...")
    strategies = []
    
    # We use random choice
    seen = set()
    while len(strategies) < N_STRATEGIES:
        lb = int(np.random.choice(POSSIBLE_LOOKBACKS))
        up = int(np.random.choice(POSSIBLE_UPPER))
        low = int(np.random.choice(POSSIBLE_LOWER))
        
        cfg = (lb, up, low)
        if cfg not in seen:
            seen.add(cfg)
            strategies.append(cfg)
            
    print("Selected Strategies (LB, Up, Low):")
    for s in strategies:
        print(s)
        
    # --------------------------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------------------------
    t0 = time.time()
    full_df = load_data_cudf(DATA_DIR)
    
    # Arrays
    prices_gpu = cp.asarray(full_df['close'].values, dtype=cp.float32)
    # We will need Pandas index for resampling
    times_pd = full_df['time'].to_pandas()
    
    print(f"Data Loaded: {len(full_df)} rows.")
    
    # --------------------------------------------------------------------------
    # 3. CALCULATE RSI (Only for selected lookbacks to save memory/time)
    # --------------------------------------------------------------------------
    # Extract unique lookbacks needed
    unique_lbs = sorted(list(set([s[0] for s in strategies])))
    # Map lookback value to row index in rsi_matrix
    lb_map = {lb: i for i, lb in enumerate(unique_lbs)}
    
    rsi_matrix = calculate_rsi_matrix_gpu(prices_gpu, unique_lbs)
    
    # --------------------------------------------------------------------------
    # 4. SIMULATE INDIVIDUAL STRATEGIES (Full History)
    # --------------------------------------------------------------------------
    # We want to generate a DataFrame of returns (or equity) for each strategy
    # to handle the weekly rebalancing logic easily in Python.
    
    print("Simulating Strategies...")
    
    # Global arrays
    # prices_gpu is (N,)
    price_diff = cp.diff(prices_gpu) # (N-1,)
    # Pad price diff to shape N (first is 0 or nan)
    price_diff_padded = cp.concatenate((cp.array([0.0], dtype=cp.float32), price_diff))
    
    # Returns Container (Strategy -> Series of Pct Returns)
    all_strat_returns = pd.DataFrame(index=times_pd) # Index matches times
    
    for i, (lb, up, low) in enumerate(strategies):
        print(f"Running Strat {i+1}/{N_STRATEGIES}: LB={lb} U={up} L={low}...")
        
        # Get RSI
        row_idx = lb_map[lb]
        rsi_arr = rsi_matrix[row_idx, :]
        
        # Logic Vectors
        # Signals
        # 1. Raw
        is_long = rsi_arr > up
        is_short = rsi_arr < low
        
        # 2. Pos State (FFill)
        # Create 'events' array: 1 (Long), 2 (Short), 0 (None).
        events = cp.zeros_like(rsi_arr, dtype=cp.int8)
        events[is_long] = 1
        events[is_short] = 2 
        
        # Accumulate max index logic
        # Since standard cupy maximum.accumulate might not be available or fail in older versions?
        # Let's use a workaround:
        # We need forward fill indices.
        # Use simple custom scanned kernel or move to CPU (but CPU is slow 3M rows).
        # Actually `maximum.accumulate` SHOULD be supported in recent CuPy.
        # If not, let's use `cupy.scan` (which doesn't exist directly like that).
        
        # Workaround: Custom Numba Kernel for State Propagation "Always In"
        # Or even simpler: Use pandas on CPU? 10 strats x 3M rows takes ~1s on CPU.
        # Let's fallback to CPU processing for the Logic Part if GPU scan fails or too complex.
        # Transfer RSI row to CPU -> Calc Signal -> Calc Returns -> Store in DF.
        
        rsi_cpu = cp.asnumpy(rsi_arr)
        
        # CPU Logic (Numba JIT or just Numpy)
        # Using Numpy for index tracking
        is_long_cpu = rsi_cpu > up
        is_short_cpu = rsi_cpu < low
        
        events_cpu = np.zeros_like(rsi_cpu, dtype=np.int8)
        events_cpu[is_long_cpu] = 1
        events_cpu[is_short_cpu] = 2
        
        # FFill indices
        idx = np.arange(len(events_cpu))
        mask = events_cpu != 0
        valid_idx = np.where(mask, idx, 0)
        # We need accumulate max. Numpy has accumulate.
        last_event_idx = np.maximum.accumulate(valid_idx)
        
        # If mask at 0 is False, valid_idx[0] is 0. last_event_idx[0] is 0.
        # Event at 0 is 0. Correct.
        # But if first event is at index 100?
        # last_event_idx[0...99] = 0.
        # events[0] = 0. Correct.
        # Wait, if events[0] is actually a signal?
        # valid_idx[0] = 0. last_event_idx[0] = 0.
        # We fetch events_cpu[0]. Correct.
        
        final_events = events_cpu[last_event_idx]
        # BUT: If NO event has occurred yet (e.g. indices 0..99),
        # last_event_idx is 0. We fetch events[0].
        # If events[0] == 0, we get 0 (Flat). Correct.
        # If events[0] != 0, we imply that signal 0 happened at index 0?
        # Yes, if signal at 0 exists, it propagates.
        # The only edge case: No signal at 0, but logic fetches index 0.
        # Since events[0] is 0, we get 0. Correct.
        
        pos_cpu = np.where(final_events == 2, -1, final_events)
        
        # Cost Logic (CPU)
        pos_prev = np.roll(pos_cpu, 1)
        pos_prev[0] = 0
        
        trade_cost = np.abs(pos_cpu - pos_prev) * (COST_PER_FLIP / 2.0)
        
        # Returns (using padded price diff from GPU -> CPU)
        diff_cpu = cp.asnumpy(price_diff_padded)
        prices_cpu = cp.asnumpy(prices_gpu)
        
        gross_ret = pos_prev * diff_cpu # Return realized at t from pos at t-1
        net_ret = gross_ret - trade_cost
        
        # Pct Return
        prices_prev = np.roll(prices_cpu, 1)
        prices_prev[0] = prices_cpu[0]
        
        pct_ret = net_ret / prices_prev
        
        all_strat_returns[f'strat_{i}'] = pct_ret
        
    print("Returns Calculation Complete.")
    
    # --------------------------------------------------------------------------
    # 5. PORTFOLIO SIMULATION (Weekly Rebalancing)
    # --------------------------------------------------------------------------
    print("Simulating Rebalanced Portfolio...")
    
    # Add 'week' column
    all_strat_returns['week'] = all_strat_returns.index.to_period('W')
    
    grouped = all_strat_returns.groupby('week')
    
    # Independent Equities (Full History, No Rebalancing)
    print("Calculating Independent Equities...")
    # (1+r).cumprod()
    # We must iterate or use cumprod (Pandas cumprod is fast)
    indep_equities = (1 + all_strat_returns.drop(columns='week')).cumprod() * INITIAL_CAPITAL
    
    # Portfolio Rebalancing Logic
    week_keys = sorted(list(grouped.groups.keys()))
    
    port_series_list = []
    
    current_capital = INITIAL_CAPITAL
    
    for w in week_keys:
        week_data = grouped.get_group(w).drop(columns='week')
        if week_data.empty: continue
        
        # Calculate growth factors for this week (Cumulative within week)
        week_growth = (1 + week_data).cumprod()
        
        # Allocation: Split capital equally
        alloc_per_strat = current_capital / N_STRATEGIES
        
        # Equity value series
        # Sum of (Alloc * Growth)
        week_value_series = (week_growth * alloc_per_strat).sum(axis=1)
        
        port_series_list.append(week_value_series)
        
        # Update capital for next week
        current_capital = week_value_series.iloc[-1]
        
    portfolio_equity_curve = pd.concat(port_series_list)
    
    # --------------------------------------------------------------------------
    # 6. PLOTTING
    # --------------------------------------------------------------------------
    print("Plotting...")
    
    plt.figure(figsize=(14, 8))
    
    # Plot Independent Strats
    colors = plt.cm.tab10(np.linspace(0, 1, N_STRATEGIES))
    
    for i, col in enumerate(indep_equities.columns):
        plt.plot(indep_equities.index, indep_equities[col], 
                 color='grey', alpha=0.3, linewidth=1, label='_nolegend_')
        
    # Plot Portfolio
    plt.plot(portfolio_equity_curve.index, portfolio_equity_curve, 
             color='black', linewidth=2.5, label='Eq Weight Portfolio (Weekly)')
             
    plt.title(f'RSI Breakout Random Portfolio (N=10) - GBPJPY\nLookback 60-30k | Cost {COST_PER_FLIP} | Weekly Rebal')
    plt.ylabel(f'Equity (Start {INITIAL_CAPITAL})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(RESULTS_DIR, "portfolio_equity.png"))
    plt.close()
    
    # Save Data
    portfolio_equity_curve.name = 'portfolio'
    portfolio_equity_curve.to_csv(os.path.join(RESULTS_DIR, "portfolio_equity.csv"))
    
    print(f"Done. Final Portfolio Value: {current_capital:.2f}")

if __name__ == "__main__":
    run_portfolio()
