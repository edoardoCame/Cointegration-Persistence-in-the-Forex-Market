import cudf
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
import os
import time

@cuda.jit
def bb_strategy_kernel(close, means, stds, std_mults, commission, results):
    lb_idx = cuda.blockIdx.x
    sm_idx = cuda.threadIdx.x
    
    if lb_idx < means.shape[0] and sm_idx < std_mults.shape[0]:
        cur_means = means[lb_idx]
        cur_stds = stds[lb_idx]
        std_mult = std_mults[sm_idx]
        
        current_pos = 0
        total_ret = 0.0
        n = len(close)
        
        for i in range(1, n):
            price = close[i]
            prev_price = close[i-1]
            m = cur_means[i]
            s = cur_stds[i]
            
            if s <= 0 or math.isnan(m) or math.isnan(s):
                continue
                
            if current_pos == 1:
                total_ret += (price - prev_price)
            elif current_pos == -1:
                total_ret += (prev_price - price)
            
            upper = m + s * std_mult
            lower = m - s * std_mult
            
            if current_pos == 0:
                if price < lower:
                    current_pos = 1
                    total_ret -= commission
                elif price > upper:
                    current_pos = -1
                    total_ret -= commission
            elif current_pos == 1:
                if price >= m:
                    current_pos = 0
                    total_ret -= commission
            elif current_pos == -1:
                if price <= m:
                    current_pos = 0
                    total_ret -= commission
                    
        results[lb_idx, sm_idx] = total_ret

def run_is_oos():
    data_path = "data/DAT_ASCII_EURGBP_M1_2025.csv"
    print(f"Loading data...")
    df = cudf.read_csv(data_path, sep=";", names=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Split 6 months IS (Jan-Jun), 6 months OOS (Jul-Dec)
    # 20250701 is the start of OOS
    df_is = df[df['timestamp'] < '20250701'].reset_index(drop=True)
    df_oos = df[df['timestamp'] >= '20250701'].reset_index(drop=True)
    
    print(f"IS samples: {len(df_is)}, OOS samples: {len(df_oos)}")
    
    # Optimization Parameters
    lookbacks = np.arange(2000, 20001, 1000)
    std_mults = np.arange(1.5, 4.1, 0.1)
    commission_cost = 0.25 * 0.0001
    
    # 1. Optimize on IS
    print("Optimizing on In-Sample data...")
    means_list = []
    stds_list = []
    for lb in lookbacks:
        means_list.append(df_is['close'].rolling(window=int(lb)).mean().fillna(np.nan))
        stds_list.append(df_is['close'].rolling(window=int(lb)).std().fillna(np.nan))
        
    close_is_gpu = cuda.to_device(df_is['close'].values)
    means_is_gpu = cuda.to_device(np.stack([m.to_numpy(na_value=np.nan) for m in means_list]))
    stds_is_gpu = cuda.to_device(np.stack([s.to_numpy(na_value=np.nan) for s in stds_list]))
    std_mults_gpu = cuda.to_device(std_mults)
    results_is_gpu = cuda.device_array((len(lookbacks), len(std_mults)), dtype=np.float64)
    
    bb_strategy_kernel[len(lookbacks), len(std_mults)](
        close_is_gpu, means_is_gpu, stds_is_gpu, std_mults_gpu, commission_cost, results_is_gpu
    )
    cuda.synchronize()
    
    results_is = results_is_gpu.copy_to_host()
    best_idx = np.unravel_index(np.argmax(results_is), results_is.shape)
    best_lb = lookbacks[best_idx[0]]
    best_sm = std_mults[best_idx[1]]
    
    print(f"\nBest IS Params: Lookback={best_lb}, Std={best_sm:.2f}, Return={results_is[best_idx]:.6f}")
    
    # 2. Backtest Best Params on IS and OOS
    def backtest(df_subset, lb, sm):
        df_subset['m'] = df_subset['close'].rolling(window=int(lb)).mean()
        df_subset['s'] = df_subset['close'].rolling(window=int(lb)).std()
        df_subset = df_subset.dropna().reset_index(drop=True)
        
        c = df_subset['close'].values.get()
        m = df_subset['m'].values.get()
        s = df_subset['s'].values.get()
        
        pos = np.zeros(len(c))
        curr = 0
        for i in range(1, len(c)):
            if curr == 0:
                if c[i] < m[i] - s[i] * sm: curr = 1
                elif c[i] > m[i] + s[i] * sm: curr = -1
            elif curr == 1:
                if c[i] >= m[i]: curr = 0
            elif curr == -1:
                if c[i] <= m[i]: curr = 0
            pos[i] = curr
            
        df_subset['position'] = pos
        df_subset['ret'] = df_subset['position'].shift(1).fillna(0) * df_subset['close'].diff().fillna(0)
        df_subset['comm'] = df_subset['position'].diff().abs().fillna(0) * commission_cost
        df_subset['net'] = df_subset['ret'] - df_subset['comm']
        df_subset['equity'] = df_subset['net'].cumsum()
        return df_subset

    print("Running OOS Validation...")
    df_is_final = backtest(df_is, best_lb, best_sm)
    df_oos_final = backtest(df_oos, best_lb, best_sm)
    
    # Final cumulative equity for OOS starts where IS ended (approximately, for visualization)
    is_last_equity = df_is_final['equity'].iloc[-1]
    df_oos_final['equity_cumulative'] = df_oos_final['equity'] + is_last_equity
    
    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(df_is_final['equity'].to_pandas(), label='In-Sample (Optimization)', color='blue')
    
    # Offset OOS indices for continuous plot
    oos_indices = np.arange(len(df_is_final), len(df_is_final) + len(df_oos_final))
    plt.plot(oos_indices, df_oos_final['equity_cumulative'].to_pandas(), label='Out-of-Sample (Validation)', color='red')
    
    plt.axvline(x=len(df_is_final), color='black', linestyle='--', label='IS/OOS Split')
    plt.title(f"IS/OOS Analysis: EURGBP 2025 (Best LB={best_lb}, Std={best_sm:.2f})")
    plt.xlabel("Minutes")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.savefig("BB backtest/is_oos_equity_curve.png")
    
    print(f"IS Return: {is_last_equity:.6f}")
    print(f"OOS Return: {df_oos_final['equity'].iloc[-1]:.6f}")
    
    # Sharpe Ratio Calculation (Annualized)
    # Annualization factor for M1 data: sqrt(252 * 24 * 60)
    ann_factor = np.sqrt(252 * 1440)
    
    def calculate_sharpe(net_returns):
        mean_ret = net_returns.mean()
        std_ret = net_returns.std()
        if std_ret == 0: return 0
        return (mean_ret / std_ret) * ann_factor

    is_sharpe = calculate_sharpe(df_is_final['net'].to_pandas())
    oos_sharpe = calculate_sharpe(df_oos_final['net'].to_pandas())
    
    print(f"\nSharpe Ratio (Annualized):")
    print(f"IS Sharpe: {is_sharpe:.2f}")
    print(f"OOS Sharpe: {oos_sharpe:.2f}")
    
    print("Plot saved as BB backtest/is_oos_equity_curve.png")

if __name__ == "__main__":
    run_is_oos()
