import cudf
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
import os
import time

# CUDA Kernel for path-dependent strategy
# One thread per (lookback, std_mult) combination
@cuda.jit
def bb_strategy_kernel(close, means, stds, std_mults, commission, results):
    lb_idx = cuda.blockIdx.x
    sm_idx = cuda.threadIdx.x
    
    if lb_idx < means.shape[0] and sm_idx < std_mults.shape[0]:
        cur_means = means[lb_idx]
        cur_stds = stds[lb_idx]
        std_mult = std_mults[sm_idx]
        
        current_pos = 0 # 0: none, 1: long, -1: short
        total_ret = 0.0
        n = len(close)
        
        for i in range(1, n):
            price = close[i]
            prev_price = close[i-1]
            m = cur_means[i]
            s = cur_stds[i]
            
            # Skip invalid values
            if s <= 0 or math.isnan(m) or math.isnan(s):
                continue
                
            # 1. Calculate returns based on PREVIOUS position
            if current_pos == 1:
                total_ret += (price - prev_price)
            elif current_pos == -1:
                total_ret += (prev_price - price)
            
            # 2. Decide position for NEXT period based on CURRENT price
            upper = m + s * std_mult
            lower = m - s * std_mult
            
            if current_pos == 0:
                if price < lower:
                    current_pos = 1
                    total_ret -= commission # Entry cost
                elif price > upper:
                    current_pos = -1
                    total_ret -= commission # Entry cost
            elif current_pos == 1:
                if price >= m:
                    current_pos = 0
                    total_ret -= commission # Exit cost
            elif current_pos == -1:
                if price <= m:
                    current_pos = 0
                    total_ret -= commission # Exit cost
                    
        results[lb_idx, sm_idx] = total_ret

def optimize():
    data_path = "data/DAT_ASCII_EURGBP_M1_2025.csv"
    print(f"Loading data from {data_path}...")
    df = cudf.read_csv(data_path, sep=";", names=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Parameters to optimize
    lookbacks = np.arange(2000, 20001, 2000) # 10 values
    std_mults = np.arange(1.5, 4.1, 0.25)    # 11 values
    
    commission_pips = 0.25
    pip_value = 0.0001
    commission_cost = commission_pips * pip_value
    
    print(f"Pre-calculating rolling stats for {len(lookbacks)} lookbacks...")
    means_list = []
    stds_list = []
    
    for lb in lookbacks:
        # Fill nulls with NaN or 0 so to_numpy doesn't complain
        # The kernel already checks for NaN
        means_list.append(df['close'].rolling(window=int(lb)).mean().fillna(np.nan))
        stds_list.append(df['close'].rolling(window=int(lb)).std().fillna(np.nan))
    
    # Move to GPU arrays (cupy-like via numba/cudf)
    close_gpu = cuda.to_device(df['close'].values)
    means_gpu = cuda.to_device(np.stack([m.to_numpy(na_value=np.nan) for m in means_list]))
    stds_gpu = cuda.to_device(np.stack([s.to_numpy(na_value=np.nan) for s in stds_list]))
    std_mults_gpu = cuda.to_device(std_mults)
    
    results_gpu = cuda.device_array((len(lookbacks), len(std_mults)), dtype=np.float64)
    
    print(f"Launching GPU kernel for {len(lookbacks) * len(std_mults)} combinations...")
    start_time = time.time()
    
    # Grid: Blocks = num_lookbacks, Threads = num_std_mults
    bb_strategy_kernel[len(lookbacks), len(std_mults)](
        close_gpu, means_gpu, stds_gpu, std_mults_gpu, commission_cost, results_gpu
    )
    cuda.synchronize()
    
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    
    results = results_gpu.copy_to_host()
    
    # Find best
    best_idx = np.unravel_index(np.argmax(results), results.shape)
    best_lookback = lookbacks[best_idx[0]]
    best_std = std_mults[best_idx[1]]
    best_ret = results[best_idx]
    
    print(f"\nBest Parameters Found:")
    print(f"Lookback: {best_lookback}")
    print(f"Std Mult: {best_std}")
    print(f"Total Return: {best_ret:.6f}")
    
    # Re-run best to plot
    print("\nRunning final backtest with best parameters...")
    df['m'] = df['close'].rolling(window=int(best_lookback)).mean()
    df['s'] = df['close'].rolling(window=int(best_lookback)).std()
    df = df.dropna().reset_index(drop=True)
    
    close = df['close'].values.get()
    m_band = df['m'].values.get()
    s_band = df['s'].values.get()
    
    n = len(close)
    pos = np.zeros(n)
    curr = 0
    for i in range(1, n):
        if curr == 0:
            if close[i] < m_band[i] - s_band[i] * best_std: curr = 1
            elif close[i] > m_band[i] + s_band[i] * best_std: curr = -1
        elif curr == 1:
            if close[i] >= m_band[i]: curr = 0
        elif curr == -1:
            if close[i] <= m_band[i]: curr = 0
        pos[i] = curr
        
    df['position'] = pos
    df['strategy_returns'] = df['position'].shift(1).fillna(0) * df['close'].diff().fillna(0)
    df['commissions'] = df['position'].diff().abs().fillna(0) * commission_cost
    df['net_returns'] = df['strategy_returns'] - df['commissions']
    df['equity_curve'] = df['net_returns'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['equity_curve'].to_pandas())
    plt.title(f"Optimized BB (LB={best_lookback}, Std={best_std}) - EURGBP 2025")
    plt.grid(True)
    plt.savefig("BB backtest/optimized_equity_curve.png")
    print("Optimized plot saved as BB backtest/optimized_equity_curve.png")

if __name__ == "__main__":
    optimize()
