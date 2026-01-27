
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
    
    # Initialize with first value (adjust=False behavior)
    avg_gain = gains[0]
    avg_loss = losses[0]
    
    # First point output
    # If min_periods > 1, this should be NaN, but we just compute the value
    # We will mask later or output it.
    # Usually RSI is 100 - ...
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
        
        # Masks for min_periods
        if i < n:
            out_rsi[lb_idx, i] = np.nan
        else:
            if avg_loss == 0:
                out_rsi[lb_idx, i] = 100.0
            else:
                rs = avg_gain / avg_loss
                out_rsi[lb_idx, i] = 100.0 - (100.0 / (1.0 + rs))

@cuda.jit
def wfo_opt_kernel(prices, rsi_matrix, configs, start_idx, end_idx, cost, out_pnl):
    """
    Optimizes logic over a time window.
    prices: (Time,)
    rsi_matrix: (N_LB, Time)
    configs: (N_Configs, 3) -> [lb_idx_in_matrix, upper, lower]
    out_pnl: (N_Configs,)
    """
    
    cfg_idx = cuda.grid(1)
    
    if cfg_idx >= configs.shape[0]:
        return
    
    # Unpack config
    lb_idx = int(configs[cfg_idx, 0])
    upper = configs[cfg_idx, 1]
    lower = configs[cfg_idx, 2]
    
    # Simulation State
    curr_pos = 0 # 0: Flat, 1: Long, -1: Short
    pnl = 0.0
    
    # Iterate through time
    # Note: We can only calculate PnL starting from start_idx + 1 (using price diff)
    # Signal calculated at i determines pos for i -> i+1
    
    # Warmup / Initial State check?
    # We assume flat at start of window for optimization comparison.
    
    # We loop from start_idx to end_idx - 1
    # For each i, we update pos based on rsi_matrix[lb_idx, i]
    # Then we apply return: pos * (prices[i+1] - prices[i])
    # If pos changes, we subtract cost.
    
    # We need to access RSI. rsi_matrix has shape (N_LB, total_time)
    
    for i in range(start_idx, end_idx):
        # Current Price and Next Price logic?
        # Usually: Signal at Close[i] -> Trade at Open[i+1] (approximated as Close[i+1] diff)
        # return = (price[i+1] - price[i]) * pos[i]
        
        # We need to stop at end_idx - 1 for returns.
        
        # Check limit for price diff
        if i >= prices.shape[0] - 1:
            break
            
        rsi_val = rsi_matrix[lb_idx, i]
        price = prices[i]
        next_price = prices[i+1]
        
        # Logic
        new_pos = curr_pos
        if rsi_val > upper:
            new_pos = 1
        elif rsi_val < lower:
            new_pos = -1
        
        # Cost Logic
        # If position changes, we pay cost.
        if new_pos != curr_pos:
            # Change Logic:
            # 0->1: change 1. 1->-1: change 2.
            # Cost = change * (cost / 2.0)
            change = abs(new_pos - curr_pos)
            pnl -= change * (cost / 2.0)
            curr_pos = new_pos
            
        # Return
        # Holding 'curr_pos' over interval i -> i+1
        step_ret = (next_price - price) * curr_pos
        pnl += step_ret
        
    out_pnl[cfg_idx] = pnl

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
        if i < n - 1:
            out_rsi[lb_idx, i] = np.nan
        else:
            if avg_loss == 0:
                out_rsi[lb_idx, i] = 100.0
            else:
                rs = avg_gain / avg_loss
                out_rsi[lb_idx, i] = 100.0 - (100.0 / (1.0 + rs))

def calculate_rsi_matrix(close_array_gpu, lookbacks):
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

def run_wfo():
    # --------------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------------
    LOOKBACKS = list(range(1440, 30001, 60))
    # Thresholds
    UPPER_THRESHOLDS = list(range(55, 96, 5)) 
    LOWER_THRESHOLDS = list(range(5, 46, 5))
    
    OPT_WINDOW_DAYS = 90
    TEST_WINDOW_DAYS = 7
    COST_PER_FLIP = 0.012 # 1.2 pips (JPY pairs)
    
    # PATHS
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate Configs Grid (Combinations)
    # List of [lb_idx, upper, lower]
    # We map lb_val to lb_idx
    import itertools
    
    # combos of (lb_idx, upper, lower)
    # lb_idx corresponds to index in LOOKBACKS list
    lb_indices = list(range(len(LOOKBACKS)))
    
    all_combos = list(itertools.product(lb_indices, UPPER_THRESHOLDS, LOWER_THRESHOLDS))
    n_configs = len(all_combos)
    
    configs_host = np.array(all_combos, dtype=np.float32) # float for GPU generic array
    configs_gpu = cp.asarray(configs_host)
    
    print(f"Total Combinations: {n_configs}")
    
    # --------------------------------------------------------------------------
    # DATA LOADING & PREP
    # --------------------------------------------------------------------------
    t0 = time.time()
    full_df = load_data_cudf(DATA_DIR)
    
    # Arrays
    times = full_df['time'].to_numpy() # CPU
    prices_gpu = cp.asarray(full_df['close'].values, dtype=cp.float32) 
    
    print(f"Data Loaded: {len(full_df)} csv rows.")
    
    # RSI Precalc
    rsi_matrix = calculate_rsi_matrix(prices_gpu, LOOKBACKS)
    print(f"RSI Matrix Shape: {rsi_matrix.shape}. Size: {rsi_matrix.nbytes / 1e9:.2f} GB")
    
    # --------------------------------------------------------------------------
    # WFO LOOP
    # --------------------------------------------------------------------------
    start_time = pd.Timestamp(full_df['time'].iloc[0])
    end_time = pd.Timestamp(full_df['time'].iloc[-1])
    
    # Time deltas in nanoseconds (int64)
    opt_window_ns = pd.Timedelta(days=OPT_WINDOW_DAYS).value
    test_week_ns = pd.Timedelta(days=TEST_WINDOW_DAYS).value
    
    # Start point: Start + Opt Window
    start_ns = (start_time + pd.Timedelta(days=OPT_WINDOW_DAYS)).value
    current_ns = start_ns
    end_ns = end_time.value
    
    # Convert times to int64 for fast search
    times_int = times.astype('int64')
    
    equity_curve = []
    param_history = []
    
    total_cum_pnl = 0.0
    current_market_pos = 0 # 0, 1, -1.  Persists across weeks.
    
    # Allocate output buffer for kernel once
    out_pnl_gpu = cp.zeros(n_configs, dtype=cp.float32)
    
    threads_per_block = 128
    blocks_per_grid = (n_configs + threads_per_block - 1) // threads_per_block
    
    print(f"Starting WFO... (Kernel Blocks: {blocks_per_grid})")
    
    step = 0
    t_start_loop = time.time()
    
    while current_ns < end_ns:
        step += 1
        
        # 1. Define Time Windows
        opt_start_ns = current_ns - opt_window_ns
        opt_end_ns = current_ns
        test_end_ns = current_ns + test_week_ns
        
        # 2. Get Indices
        idx_opt_start = np.searchsorted(times_int, opt_start_ns)
        idx_opt_end = np.searchsorted(times_int, opt_end_ns)
        idx_test_end = np.searchsorted(times_int, test_end_ns)
        
        if idx_opt_end >= len(prices_gpu):
            break
            
        # 3. RUN OPTIMIZATION (In-Sample)
        # We run on [idx_opt_start, idx_opt_end]
        
        # Reset output
        out_pnl_gpu.fill(0.0) # Reset PnL
        
        wfo_opt_kernel[blocks_per_grid, threads_per_block](
            prices_gpu, 
            rsi_matrix, 
            configs_gpu, 
            idx_opt_start, 
            idx_opt_end, 
            COST_PER_FLIP, 
            out_pnl_gpu
        )
        
        # Find Best
        # Use simple argmax on the results array
        best_pnl_idx = int(cp.argmax(out_pnl_gpu))
        best_pnl_val = float(out_pnl_gpu[best_pnl_idx])
        
        # Retrieve Params
        # configs_host[best_pnl_idx] -> [lb_idx, upper, lower]
        best_cfg = configs_host[best_pnl_idx]
        best_lb_idx = int(best_cfg[0])
        best_lb = LOOKBACKS[best_lb_idx]
        best_upper = float(best_cfg[1])
        best_lower = float(best_cfg[2])
        
        # Log
        curr_date = pd.to_datetime(current_ns)
        if step % 20 == 0:
            print(f"Week {step} | {curr_date.date()} | Best: LB={best_lb} U={best_upper} L={best_lower} | IS_PnL={best_pnl_val:.4f}")
            
        param_history.append({
            'date': curr_date,
            'lookback': best_lb,
            'upper': best_upper,
            'lower': best_lower,
            'is_pnl': best_pnl_val
        })
        
        # 4. RUN TEST (Out-Of-Sample)
        # Apply best params to [idx_opt_end, idx_test_end]
        # Must maintain continuity of `current_market_pos`
        
        if idx_test_end > idx_opt_end:
            # We run this on CPU/Host for simplicity (logic is sequential and short)
            # Fetch data slices
            test_rsi_slice = rsi_matrix[best_lb_idx, idx_opt_end:idx_test_end] # GPU view
            test_price_slice = prices_gpu[idx_opt_end:idx_test_end] # GPU view
            
            # Copy to Host
            rsi_h = cp.asnumpy(test_rsi_slice)
            price_h = cp.asnumpy(test_price_slice)
            
            # Loop
            week_pnl = 0.0
            
            # We iterate prices. 
            # Logic:
            # At step i (relative to slice):
            # Check RSI[i]. Update Pos.
            # Return = Pos * (Price[i+1] - Price[i])
            # Wait, Price[i+1] requires us to have N prices.
            
            n_points = len(price_h)
            
            for i in range(n_points - 1): # Can't calc return for last point without next price
                curr_price = price_h[i]
                next_price = price_h[i+1]
                rsi_val = rsi_h[i]
                
                # Signal Logic
                new_pos = current_market_pos
                if rsi_val > best_upper:
                    new_pos = 1
                elif rsi_val < best_lower:
                    new_pos = -1
                
                # Cost
                if new_pos != current_market_pos:
                    change = abs(new_pos - current_market_pos)
                    cost = change * (COST_PER_FLIP / 2.0)
                    week_pnl -= cost
                    current_market_pos = new_pos
                
                # Return
                ret = (next_price - curr_price) * current_market_pos
                week_pnl += ret
                
            total_cum_pnl += week_pnl
            
            date_val = times[idx_test_end] if idx_test_end < len(times) else times[-1]
            equity_curve.append({
                'date': pd.to_datetime(date_val),
                'equity': total_cum_pnl
            })
            
        # Advance
        current_ns += test_week_ns

    print(f"WFO Complete. Total Time: {time.time() - t_start_loop:.2f}s")
    
    # --------------------------------------------------------------------------
    # SAVE & PLOT
    # --------------------------------------------------------------------------
    
    # Equity
    eq_df = pd.DataFrame(equity_curve)
    eq_df.to_csv(os.path.join(RESULTS_DIR, "equity_curve.csv"), index=False)
    
    # Params
    param_df = pd.DataFrame(param_history)
    param_df.to_csv(os.path.join(RESULTS_DIR, "param_history.csv"), index=False)
    
    try:
        # Plot 1: Equity
        plt.figure(figsize=(12, 6))
        plt.plot(eq_df['date'], eq_df['equity'], label='WFO Equity')
        plt.title(f'RSI Breakout WFO (GBPJPY) | Final: {total_cum_pnl:.5f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "equity_curve.png"))
        plt.close()
        
        # Plot 2: Params
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        ax1.plot(param_df['date'], param_df['lookback'], 'b-', label='Lookback', alpha=0.6)
        ax1.set_ylabel('Lookback', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(param_df['date'], param_df['upper'], 'g--', label='Upper', alpha=0.6)
        ax2.plot(param_df['date'], param_df['lower'], 'r--', label='Lower', alpha=0.6)
        ax2.set_ylabel('Threshold', color='k')
        
        plt.title('Parameter Stability')
        plt.savefig(os.path.join(RESULTS_DIR, "params.png"))
        plt.close()
        
    except Exception as e:
        print(f"Plotting Error: {e}")

if __name__ == "__main__":
    run_wfo()
