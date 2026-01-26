import cudf
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# 1. CUDA Kernel for Grid Search (Optimization)
@cuda.jit
def bb_grid_search_kernel(close, means, stds, std_mults, commission, results):
    lb_idx = cuda.blockIdx.x
    sm_idx = cuda.threadIdx.x
    
    if lb_idx < means.shape[0] and sm_idx < std_mults.shape[0]:
        cur_means = means[lb_idx]
        cur_stds = stds[lb_idx]
        std_mult = std_mults[sm_idx]
        
        current_pos = 0 
        total_ret = 0.0
        n_trades = 0
        n = len(close)
        
        for i in range(1, n):
            price = close[i]
            prev_price = close[i-1]
            m = cur_means[i]
            s = cur_stds[i]
            
            if s <= 0 or math.isnan(m) or math.isnan(s): continue
                
            if current_pos == 1: total_ret += (price - prev_price)
            elif current_pos == -1: total_ret += (prev_price - price)
            
            upper = m + s * std_mult
            lower = m - s * std_mult
            
            if current_pos == 0:
                if price < lower:
                    current_pos = 1
                    total_ret -= commission
                    n_trades += 1
                elif price > upper:
                    current_pos = -1
                    total_ret -= commission
                    n_trades += 1
            elif current_pos == 1:
                if price >= m:
                    current_pos = 0
                    total_ret -= commission
            elif current_pos == -1:
                if price <= m:
                    current_pos = 0
                    total_ret -= commission
                    
        results[lb_idx, sm_idx, 0] = total_ret
        results[lb_idx, sm_idx, 1] = float(n_trades)

# 2. CUDA Kernel for Final Path (Equity Curve generation on GPU)
@cuda.jit
def bb_final_path_kernel(close, m, s, std_mult, commission, net_returns):
    # Single thread execution for the specific best path
    if cuda.grid(1) == 0:
        current_pos = 0
        for i in range(1, len(close)):
            price = close[i]
            prev_price = close[i-1]
            
            # 1. Calculate returns
            ret = 0.0
            if current_pos == 1: ret = (price - prev_price)
            elif current_pos == -1: ret = (prev_price - price)
            
            # 2. Position logic
            upper = m[i] + s[i] * std_mult
            lower = m[i] - s[i] * std_mult
            
            comm = 0.0
            if current_pos == 0:
                if price < lower:
                    current_pos = 1
                    comm = commission
                elif price > upper:
                    current_pos = -1
                    comm = commission
            elif current_pos == 1:
                if price >= m[i]:
                    current_pos = 0
                    comm = commission
            elif current_pos == -1:
                if price <= m[i]:
                    current_pos = 0
                    comm = commission
            
            net_returns[i] = ret - comm

def run_is_oos_gpu_only(lookbacks=None, std_mults=None):
    data_path = "data/DAT_ASCII_EURGBP_M1_2025.csv"
    df = cudf.read_csv(data_path, sep=";", names=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Convert timestamp to datetime for better plotting
    df['timestamp'] = cudf.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
    
    split_idx = int(len(df) * 0.7)
    df_is = df.iloc[:split_idx].reset_index(drop=True)
    df_oos = df.iloc[split_idx:].reset_index(drop=True)
    
    # Parameters
    if lookbacks is None:
        lookbacks = np.arange(1000, 20001, 1000)
    if std_mults is None:
        std_mults = np.arange(1.5, 5.1, 0.1)
        
    # Average round-trip is 1 pip, so 0.5 per entry and 0.5 per exit
    commission_pips = 0.5
    pip_value = 0.0001
    commission_cost = commission_pips * pip_value
    
    # Pre-calculate all stats on GPU using cudf
    from tqdm import tqdm
    print("Calculating rolling stats on GPU...")
    means_is_list = [df_is['close'].rolling(int(lb)).mean().fillna(np.nan).to_numpy() for lb in tqdm(lookbacks, desc="Calculating Means")]
    stds_is_list = [df_is['close'].rolling(int(lb)).std().fillna(np.nan).to_numpy() for lb in tqdm(lookbacks, desc="Calculating Stds")]
    
    close_is_gpu = cuda.to_device(df_is['close'].values)
    means_is_gpu = cuda.to_device(np.stack(means_is_list))
    stds_is_gpu = cuda.to_device(np.stack(stds_is_list))
    std_mult_gpu = cuda.to_device(std_mults)
    results_gpu = cuda.device_array((len(lookbacks), len(std_mults), 2), dtype=np.float64)
    
    print(f"Launching GPU Grid Search...")
    bb_grid_search_kernel[len(lookbacks), len(std_mults)](
        close_is_gpu, means_is_gpu, stds_is_gpu, std_mult_gpu, commission_cost, results_gpu
    )
    cuda.synchronize()
    
    results = results_gpu.copy_to_host()
    total_returns_is = results[:, :, 0]
    trades_is = results[:, :, 1]
    
    # Revert to Total Return optimization with a basic safety filter
    masked_returns = np.where(trades_is >= 20, total_returns_is, -1e9)
    
    best_idx = np.unravel_index(np.argmax(masked_returns), masked_returns.shape)
    best_lb, best_sm = lookbacks[best_idx[0]], std_mults[best_idx[1]]
    
    print(f"Best Params: LB={best_lb}, Std={best_sm:.2f}")

    print("Generating Continuous Equity Curve on GPU...")
    # Calculate stats on full dataframe to avoid warm-up gaps
    m_full = df['close'].rolling(int(best_lb)).mean().fillna(np.nan)
    s_full = df['close'].rolling(int(best_lb)).std().fillna(np.nan)
    
    d_close_full = cuda.to_device(df['close'].values)
    d_m_full = cuda.to_device(m_full.values)
    d_s_full = cuda.to_device(s_full.values)
    # Correctly initialize memory to zero
    d_net_full = cuda.to_device(np.zeros(len(df), dtype=np.float64))
    
    bb_final_path_kernel[1, 1](d_close_full, d_m_full, d_s_full, best_sm, commission_cost, d_net_full)
    cuda.synchronize()
    
    full_equity = cudf.Series(d_net_full).cumsum()
    
    # --- STATS CALCULATION ---
    # IS Stats (from Grid Search results)
    is_ret = results[best_idx[0], best_idx[1], 0]
    is_trades = int(results[best_idx[0], best_idx[1], 1])
    is_avg_pips = (is_ret / is_trades / pip_value) if is_trades > 0 else 0
    
    # OOS Stats
    d_net_array = d_net_full.copy_to_host()
    oos_net_returns = d_net_array[split_idx:]
    oos_ret = full_equity.iloc[-1] - full_equity.iloc[split_idx]
    
    # Re-run trade count for OOS
    current_pos = 0
    oos_trades = 0
    c_oos = df['close'].values[split_idx:].get()
    m_oos = m_full.values[split_idx:]
    s_oos = s_full.values[split_idx:]
    for i in range(1, len(c_oos)):
        upper = m_oos[i] + s_oos[i] * best_sm
        lower = m_oos[i] - s_oos[i] * best_sm
        if current_pos == 0:
            if c_oos[i] < lower or c_oos[i] > upper:
                current_pos = 1 if c_oos[i] < lower else -1
                oos_trades += 1
        elif current_pos == 1:
            if c_oos[i] >= m_oos[i]: current_pos = 0
        elif current_pos == -1:
            if c_oos[i] <= m_oos[i]: current_pos = 0
            
    oos_avg_pips = (oos_ret / oos_trades / pip_value) if oos_trades > 0 else 0

    stats_text = (
        f"Strategy Statistics (EURGBP 2025)\n"
        f"Parameters: Lookback={best_lb}, Std={best_sm:.2f}\n"
        f"Commission: {commission_pips*2} pips round-trip\n"
        f"{'-'*30}\n"
        f"IN-SAMPLE (Optimization):\n"
        f"  Total Return: {is_ret:.6f} ({is_ret/pip_value:.1f} pips)\n"
        f"  Total Trades: {is_trades}\n"
        f"  Avg Profit per Trade: {is_avg_pips:.2f} pips\n"
        f"{'-'*30}\n"
        f"OUT-OF-SAMPLE (Validation):\n"
        f"  Total Return: {oos_ret:.6f} ({oos_ret/pip_value:.1f} pips)\n"
        f"  Total Trades: {oos_trades}\n"
        f"  Avg Profit per Trade: {oos_avg_pips:.2f} pips\n"
    )
    
    with open("BB backtest/strategy_stats.txt", "w") as f:
        f.write(stats_text)
    print(stats_text)

    return {
        'lookbacks': lookbacks,
        'std_mults': std_mults,
        'results': results,
        'full_equity': full_equity,
        'best_lb': best_lb,
        'best_sm': best_sm,
        'timestamps': df['timestamp'],
        'split_idx': split_idx,
        'is_trades': is_trades,
        'oos_trades': oos_trades,
        'is_avg_pips': is_avg_pips,
        'oos_avg_pips': oos_avg_pips
    }

if __name__ == "__main__":
    res = run_is_oos_gpu_only()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    ts = res['timestamps'].to_pandas()
    eq = res['full_equity'].to_pandas()
    split = res['split_idx']
    
    plt.plot(ts.iloc[:split], eq.iloc[:split], label=f"IS (Avg: {res['is_avg_pips']:.1f} pips)", color='blue')
    plt.plot(ts.iloc[split:], eq.iloc[split:], label=f"OOS (Avg: {res['oos_avg_pips']:.1f} pips)", color='red')
    
    plt.axvline(x=ts.iloc[split], color='black', linestyle='--', alpha=0.5)
    plt.title(f"BB Backtest: LB={res['best_lb']}, Std={res['best_sm']:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("BB backtest/is_oos_equity.png")
    
    # 3D Surface (Interactive HTML)
    fig_html = go.Figure(data=[go.Surface(z=res['results'][:,:,0], x=res['std_mults'], y=res['lookbacks'])])
    fig_html.write_html("BB backtest/is_optimization_3d.html")
    
    # 3D Surface (Beautiful Static PNG)
    print("Generating high-quality static 3D plot...")
    from mpl_toolkits.mplot3d import Axes3D
    fig_static = plt.figure(figsize=(12, 8), dpi=200)
    ax = fig_static.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(res['std_mults'], res['lookbacks'])
    Z = res['results'][:, :, 0]
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    
    ax.set_title('BB Optimization Surface - EURGBP 2025', fontsize=15, pad=20)
    ax.set_xlabel('Std Mult', fontsize=12)
    ax.set_ylabel('Lookback', fontsize=12)
    ax.set_zlabel('Net Return', fontsize=12)
    
    fig_static.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    ax.view_init(elev=30, azim=225)
    plt.tight_layout()
    plt.savefig("BB backtest/is_optimization_3d_static.png")
    
    print("All tasks completed on GPU.")

if __name__ == "__main__":
    run_is_oos_gpu_only()
