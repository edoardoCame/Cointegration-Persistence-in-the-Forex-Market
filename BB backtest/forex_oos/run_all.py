import sys
import os
import glob
import time
import cudf
import numpy as np
import cupy as cp
import pandas as pd
from numba import cuda
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add scripts to path to import kernels
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "BB backtest/scripts")))
try:
    from bb_gpu_lib import bb_batch_grid_search_kernel, bb_wfa_oos_kernel
except ImportError:
    # Fallback/Debug if running from different location
    sys.path.append("/mnt/ssd2/DARWINEX_Mission/BB backtest/scripts")
    from bb_gpu_lib import bb_batch_grid_search_kernel, bb_wfa_oos_kernel

def get_pip_scale(pair_name):
    if "JPY" in pair_name:
        return 0.01
    return 0.0001

def load_data(filepath):
    # Read CSV using cudf
    # Format: 20250101 170000;1.035030;...
    # Col names: Date, Open, High, Low, Close, Volume (implied)
    # We need to handle the semicolon delimiter
    try:
        df = cudf.read_csv(filepath, sep=';', header=None, names=['DateStr', 'Open', 'High', 'Low', 'Close', 'Vol'])
        
        # Parse datetime
        # DateStr format '20250101 170000'
        df['Date'] = cudf.to_datetime(df['DateStr'], format='%Y%m%d %H%M%S')
        
        # Drop duplicates based on Date (keep last)
        df = df.sort_values('Date')
        df = df.drop_duplicates(subset=['Date'], keep='last')
        
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def run_is_oos_backtest():
    files = sorted(glob.glob("data/DAT_ASCII_*.csv"))
    # Filter out USDMXN explicitly as requested
    files = [f for f in files if "USDMXN" not in f]

    if not files:
        print("No data files found in data/")
        return

    print(f"Found {len(files)} pairs.")
    
    # Params from montecarlo_sim.py
    lookbacks = np.arange(60, 20001, 1000).astype(np.float64)
    std_mults = np.arange(1.5, 7.1, 0.25).astype(np.float64)
    
    all_oos_curves = {}
    pair_stats = []

    for filepath in tqdm(files, desc="Processing Pairs"):
        pair_name = os.path.basename(filepath).split('_')[2] # DAT_ASCII_EURUSD_... -> EURUSD
        pip_scale = get_pip_scale(pair_name)
        commission = 0.75 * pip_scale
        
        df = load_data(filepath)
        if df is None or len(df) < 5000:
            continue
            
        prices = df['Close'].values # Cp array
        n_samples = len(prices)
        split_idx = int(n_samples * 0.7)
        
        # --- IN SAMPLE: Grid Search ---
        # Prepare for Kernel
        # We process 1 path, but kernel expects batch. We fake batch=1.
        all_paths_gpu = cp.expand_dims(prices, axis=0) # (1, n_samples)
        d_win_indices = cuda.to_device(np.array([[0, split_idx]])) # (1, 2)
        d_results = cuda.device_array((1, len(lookbacks), len(std_mults), 2), dtype=np.float64)
        
        # Launch Grid Search
        # Block dims: (1, n_lbs), Grid dims: n_sms
        # Just use (1, len(lbs)) for blocks Y, and len(sms) for threads X
        # Since lbs=20, sms=23, 20*23 = ~460 threads total if allowed?
        # The kernel uses:
        # w_idx = blockIdx.x
        # lb_idx = blockIdx.y
        # sm_idx = threadIdx.x 
        # So we launch grid (1, len(lookbacks)), block(len(std_mults))
        
        bb_batch_grid_search_kernel[(1, len(lookbacks)), len(std_mults)](
            all_paths_gpu, d_win_indices, cuda.to_device(lookbacks), cuda.to_device(std_mults), commission, d_results
        )
        cuda.synchronize()
        
        # Find Best Params
        res = d_results.copy_to_host()[0] # (n_lbs, n_sms, 2)
        # Filter trades >= 20
        masked_ret = np.where(res[:,:,1] >= 20, res[:,:,0], -1e9)
        if np.max(masked_ret) == -1e9:
            # Fallback if no params satisfy condition (very unlikely)
            masked_ret = res[:,:,0]
            
        idx = np.unravel_index(np.argmax(masked_ret), masked_ret.shape)
        best_lb = lookbacks[idx[0]]
        best_sm = std_mults[idx[1]]
        best_profit = res[idx[0], idx[1], 0]
        
        # --- OUT OF SAMPLE: Backtest ---
        # Using WFA kernel for convenience to get curve
        # week_params: [[best_lb, best_sm]]
        # week_indices: [[split_idx, n_samples]]
        
        week_params = cuda.to_device(np.array([[best_lb, best_sm]], dtype=np.float64))
        week_indices = cuda.to_device(np.array([[split_idx, n_samples]], dtype=np.float64))
        d_net_returns = cuda.device_array(n_samples, dtype=np.float64)
        
        # Initialize net_returns to 0
        d_net_returns[:] = 0.0
        
        bb_wfa_oos_kernel[1, 1](prices, week_params, week_indices, commission, d_net_returns)
        cuda.synchronize()
        
        # Process Results
        net_returns = d_net_returns.copy_to_host()
        oos_returns_pips = net_returns[split_idx:] / pip_scale
        oos_dates = df['Date'].to_pandas()[split_idx:]
        
        # Create Series
        s_res = pd.Series(oos_returns_pips, index=oos_dates, name=pair_name)
        all_oos_curves[pair_name] = s_res
        
        # Loop over, cleanup
        del df, all_paths_gpu, d_results, d_net_returns
        cp._default_memory_pool.free_all_blocks()
        
        pair_stats.append({
            'Pair': pair_name,
            'Best LB': best_lb,
            'Best SM': best_sm,
            'IS Profit (Pips)': best_profit / pip_scale,
            'OOS Profit (Pips)': np.sum(oos_returns_pips)
        })

    # --- AGGREGATION & PLOTTING ---
    if not all_oos_curves:
        print("No results generated.")
        return

    # Combine into DataFrame (Outer Join on Time)
    portfolio_df = pd.concat(all_oos_curves.values(), axis=1, keys=all_oos_curves.keys())
    portfolio_df = portfolio_df.fillna(0.0)
    
    # Sort index
    portfolio_df = portfolio_df.sort_index()
    
    # Cumulative Sums
    cum_returns = portfolio_df.cumsum()
    
    # Portfolio (Equal Weight - Sum of Pips)
    cum_returns['Portfolio'] = cum_returns.sum(axis=1)
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Plot individual pairs (faint)
    for col in cum_returns.columns:
        if col != 'Portfolio':
            plt.plot(cum_returns.index, cum_returns[col], alpha=0.3, linewidth=1, label='_nolegend_')
            
    # Plot Portfolio
    plt.plot(cum_returns.index, cum_returns['Portfolio'], color='black', linewidth=2.5, label='Portfolio (Sum Pips)')
    
    plt.title('Forex OOS Backtest - Equal Weight Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Net Pips')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_dir = "BB backtest/forex_oos"
    plt.savefig(f"{output_dir}/portfolio_oos.png")
    print(f"Saved plot to {output_dir}/portfolio_oos.png")
    
    # Save Stats
    stats_df = pd.DataFrame(pair_stats)
    stats_df.to_csv(f"{output_dir}/is_oos_stats.csv", index=False)
    print("Saved stats to is_oos_stats.csv")
    
    print("\nTop Performers (OOS):")
    print(stats_df.sort_values('OOS Profit (Pips)', ascending=False).head(5))

    print("\nLikely losers:")
    print(stats_df.sort_values('OOS Profit (Pips)', ascending=True).head(5))

if __name__ == "__main__":
    run_is_oos_backtest()
