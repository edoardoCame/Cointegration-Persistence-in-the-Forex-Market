import cudf
import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
import os
import time
import sys

# Import shared kernels
sys.path.append(os.path.join(os.getcwd(), "BB backtest/scripts"))
from bb_gpu_lib import bb_batch_grid_search_kernel, bb_wfa_oos_kernel

def run_walk_forward():
    data_path = "data/DAT_ASCII_EURGBP_M1_2025.csv"
    print("Loading data...")
    df = cudf.read_csv(data_path, sep=";", names=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = cudf.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
    
    # Week boundaries
    pdf_dates = df['timestamp'].to_pandas()
    df['week_id'] = (pdf_dates.dt.year.astype(str) + "_" + pdf_dates.dt.isocalendar().week.astype(str)).values
    week_ids = df['week_id'].unique().to_arrow().to_pylist()
    week_bounds = [(df.index[df['week_id'] == wid].min(), df.index[df['week_id'] == wid].max() + 1) for wid in week_ids]
    
    train_size = 12
    wfa_windows = [{'train': (week_bounds[i-train_size][0], week_bounds[i-1][1]), 'test': week_bounds[i], 'label': week_ids[i]} 
                   for i in range(train_size, len(week_bounds))]
    
    # Params
    lookbacks = np.arange(60, 20001, 300).astype(np.float64)
    std_mults = np.arange(1.5, 8.1, 0.25).astype(np.float64)
    commission = 0.5 * 0.0001
    
    # Grid Search
    d_prices = cuda.to_device(df['close'].values)
    d_win_indices = cuda.to_device(np.array([w['train'] for w in wfa_windows]))
    d_results = cuda.device_array((len(wfa_windows), len(lookbacks), len(std_mults), 2), dtype=np.float64)
    
    print(f"Grid Search on {len(wfa_windows)} windows...")
    bb_batch_grid_search_kernel[(len(wfa_windows), len(lookbacks)), len(std_mults)](
        d_prices, d_win_indices, cuda.to_device(lookbacks), cuda.to_device(std_mults), commission, d_results
    )
    cuda.synchronize()
    
    # Best params & OOS
    results = d_results.copy_to_host()
    best_params = []
    for i in range(len(wfa_windows)):
        res = results[i]
        masked = np.where(res[:,:,1] >= 40, res[:,:,0], -1e9)
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        best_params.append([lookbacks[idx[0]], std_mults[idx[1]]])
    
    d_net_returns = cuda.to_device(np.zeros(len(df), dtype=np.float64))
    bb_wfa_oos_kernel[1, 1](d_prices, cuda.to_device(np.array(best_params)), cuda.to_device(np.array([w['test'] for w in wfa_windows])), commission, d_net_returns)
    cuda.synchronize()
    
    df['equity'] = cudf.Series(d_net_returns).cumsum()
    
    # Plots
    res_dir = "BB backtest/results/walk_forward"
    os.makedirs(res_dir, exist_ok=True)
    wfa_start = wfa_windows[0]['test'][0]
    
    # Convert to pandas for plotting
    plot_ts = df['timestamp'].iloc[wfa_start:].to_pandas()
    plot_equity = (df['equity'].iloc[wfa_start:] - df['equity'].iloc[wfa_start]).to_pandas() / 0.0001
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_ts, plot_equity, color='green')
    plt.title("WFA Equity (OOS Pips)")
    plt.ylabel("Pips")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{res_dir}/wfa_equity.png")
    
    # Save Stats
    final_pips = plot_equity.iloc[-1]
    with open(f"{res_dir}/wfa_stats.txt", "w") as f:
        f.write(f"Walk-Forward Analysis Results\n")
        f.write(f"Total OOS Profit: {final_pips:.2f} pips\n")
        f.write(f"Windows: {len(wfa_windows)}\n")
        f.write(f"Train: 12 weeks, Test: 1 week\n")
    
    print(f"WFA Done. Profit: {final_pips:.1f} pips")
    print(f"Results saved in: {res_dir}")

if __name__ == "__main__":
    run_walk_forward()
