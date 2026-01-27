
import os
import glob
import time
import math
import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

# Data & Output Config
DATA_DIR = "/mnt/ssd2/DARWINEX_Mission/data/"
OUTPUT_DIR = "./donchian breakout/backtest_eurusd/results"
COST_TOTAL = 0.00008  # 0.8 pips
COST_HALF = COST_TOTAL / 2.0

@cuda.jit
def backtest_kernel(opens, highs, lows, closes, 
                    upper_bands, lower_bands, atrs, 
                    hours,
                    atr_multipliers,
                    min_atr_thresholds,
                    results_pnl, results_trades, results_equity):
    
    # Grid: Flattened (Window x Mult x Threshold)
    idx = cuda.grid(1)
    
    num_windows = upper_bands.shape[1]
    num_mults = atr_multipliers.shape[0]
    num_thresh = min_atr_thresholds.shape[0]
    
    total_combos = num_windows * num_mults * num_thresh
    if idx >= total_combos:
        return

    # Decode Index
    t_idx = idx % num_thresh
    tmp = idx // num_thresh
    m_idx = tmp % num_mults
    w_idx = tmp // num_mults
    
    atr_mult = atr_multipliers[m_idx]
    min_atr = min_atr_thresholds[t_idx]
    
    n_bars = closes.shape[0]
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    equity = 0.0
    trade_count = 0
    
    # Loop bars
    for i in range(1, n_bars - 1):
        curr_close = closes[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_open_next = opens[i+1] # Execution
        curr_hour_next = hours[i+1] # Check hour for ENTRY

        # Indicators for this window
        upper = upper_bands[i, w_idx]
        lower = lower_bands[i, w_idx]
        current_atr = atrs[i, w_idx] # ATR[i] known at Open[i+1]
        
        # 1. Check Stops/Exits on current bar
        if position == 1:
            if curr_low < stop_loss:
                # Check for gap
                exit_px = stop_loss
                if opens[i] < stop_loss:
                    exit_px = opens[i]
                equity += (exit_px - entry_price) - COST_TOTAL
                position = 0
                trade_count += 1
        elif position == -1:
             if curr_high > stop_loss:
                exit_px = stop_loss
                if opens[i] > stop_loss:
                    exit_px = opens[i]
                equity += (entry_price - exit_px) - COST_TOTAL
                position = 0
                trade_count += 1

        # 2. Record Equity
        unrealized = 0.0
        if position == 1:
             unrealized = (curr_close - entry_price) - COST_HALF
        elif position == -1:
             unrealized = (entry_price - curr_close) - COST_HALF
        results_equity[i, idx] = equity + unrealized

        # 3. Generate Signals
        if math.isnan(upper) or math.isnan(lower) or math.isnan(current_atr):
            continue
            
        signal = 0
        if curr_close > upper:
            signal = 1
        elif curr_close < lower:
            signal = -1
            
        # 4. Entry Logic
        if signal != 0:
            # Check Filters: Hour + Volatility
            # Hour Filter: ALL HOURS allowed (User Request)
            time_ok = True
            
            # Volatility Filter: ATR > Threshold
            vol_ok = (current_atr > min_atr)
            
            can_trade = time_ok and vol_ok
            
            # Reversal / New Entry
            if position != signal:
                # Close existing if active (Reversal logic)
                if position != 0:
                    exit_px = curr_open_next
                    pnl = (exit_px - entry_price) if position == 1 else (entry_price - exit_px)
                    equity += pnl - COST_TOTAL
                    trade_count += 1
                    position = 0
                
                # Enter new if allowed
                if can_trade:
                    position = signal
                    entry_price = curr_open_next
                    dist = current_atr * atr_mult
                    if position == 1:
                        stop_loss = entry_price - dist
                    else:
                        stop_loss = entry_price + dist

    # Save Final PnL and Trades
    results_pnl[idx] = equity
    results_trades[idx] = trade_count

def load_data():
    pattern = os.path.join(DATA_DIR, "DAT_ASCII_EURUSD_M1_*.csv")
    files = sorted(glob.glob(pattern))
    print(f"Loading {len(files)} files...")
    dfs = []
    for f in files:
        df_y = cudf.read_csv(f, sep=';', header=None, names=['DateTime','Open','High','Low','Close','Volume'])
        # Optimize date parsing by splitting strings manually if needed, or trust cudf speed
        df_y['DateTime'] = cudf.to_datetime(df_y['DateTime'], format='%Y%m%d %H%M%S')
        dfs.append(df_y)
    df = cudf.concat(dfs, ignore_index=True)
    df = df.sort_values('DateTime').reset_index(drop=True)
    return df

def run_strategy():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print("Loading Data...")
    df = load_data()
    n_bars = len(df)
    
    # PARAMETER RANGES (Simplified/Focused)
    windows = [2000, 4000, 6000, 8000, 10000] 
    atr_mults = [4.0, 5.0, 6.0, 7.0]
    min_atrs = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005] # Expanded range: 0 to 5 pips
    
    print(f"Combos: {len(windows)} Win x {len(atr_mults)} Mult x {len(min_atrs)} Vol = {len(windows)*len(atr_mults)*len(min_atrs)}")
    
    # Pre-calculate Indicators on host/gpu
    print("Calc Indicators...")
    uppers_list, lowers_list, atrs_list = [], [], []
    
    d_high = df['High']
    d_low = df['Low']
    d_close = df['Close']
    
    for w in windows:
        # Donchian
        uppers_list.append(d_high.rolling(w).max().shift(1).fillna(0.0))
        lowers_list.append(d_low.rolling(w).min().shift(1).fillna(0.0))
        
        # ATR (Simple Rolling Mean of TR)
        prev_close = d_close.shift(1)
        tr1 = d_high - d_low
        tr2 = (d_high - prev_close).abs()
        tr3 = (prev_close - d_low).abs()
        tr = cudf.DataFrame({'a':tr1,'b':tr2,'c':tr3}).max(axis=1)
        atrs_list.append(tr.rolling(14).mean().shift(1).fillna(0.0))
        
    # Stack to Cupy
    d_opens = cp.asarray(df['Open'].values)
    d_highs = cp.asarray(df['High'].values)
    d_lows = cp.asarray(df['Low'].values)
    d_closes = cp.asarray(df['Close'].values)
    d_hours = cp.asarray(df['DateTime'].dt.hour.values).astype(cp.int8)
    
    d_upper = cp.stack([cp.asarray(s.values) for s in uppers_list], axis=1)
    d_lower = cp.stack([cp.asarray(s.values) for s in lowers_list], axis=1)
    d_atrs = cp.stack([cp.asarray(s.values) for s in atrs_list], axis=1)
    
    d_atr_mults = cp.asarray(atr_mults)
    d_min_atrs = cp.asarray(min_atrs)
    
    n_strats = len(windows) * len(atr_mults) * len(min_atrs)
    
    # Output arrays
    d_res_pnl = cp.zeros(n_strats, dtype=cp.float64)
    d_res_trd = cp.zeros(n_strats, dtype=cp.int32)
    d_res_eq = cp.zeros((n_bars, n_strats), dtype=cp.float64)
    
    # Launch
    threads = 128
    blocks = (n_strats + threads - 1) // threads
    
    print("Running CUDA Kernel...")
    t0 = time.time()
    backtest_kernel[blocks, threads](
        d_opens, d_highs, d_lows, d_closes,
        d_upper, d_lower, d_atrs,
        d_hours,
        d_atr_mults,
        d_min_atrs,
        d_res_pnl, d_res_trd, d_res_eq
    )
    cuda.synchronize()
    print(f"Done in {time.time()-t0:.2f}s")
    
    # Analyze Results
    res_pnl = d_res_pnl.get()
    res_trd = d_res_trd.get()
    res_eq = d_res_eq.get()
    
    best_idx = np.argmax(res_pnl)
    
    # Decode Best
    # idx = w * (Nm * Nt) + m * Nt + t
    num_thresh = len(min_atrs)
    num_mults = len(atr_mults)
    
    t_idx = best_idx % num_thresh
    tmp = best_idx // num_thresh
    m_idx = tmp % num_mults
    w_idx = tmp // num_mults
    
    print(f"\nBEST RESULT:")
    print(f"PnL: {res_pnl[best_idx]:.4f}")
    print(f"Trades: {res_trd[best_idx]}")
    print(f"Window: {windows[w_idx]}")
    print(f"ATR Mul: {atr_mults[m_idx]}")
    print(f"Min Vol: {min_atrs[t_idx]}")
    
    # Save Stats
    import pandas as pd
    stats_list = []
    for i in range(n_strats):
        ti = i % num_thresh
        tmpi = i // num_thresh
        mi = tmpi % num_mults
        wi = tmpi // num_mults
        stats_list.append({
            "Window": windows[wi],
            "ATR_Mult": atr_mults[mi],
            "Min_ATR": min_atrs[ti],
            "PnL": res_pnl[i],
            "Trades": res_trd[i]
        })
    pd.DataFrame(stats_list).sort_values("PnL", ascending=False).to_csv(f"{OUTPUT_DIR}/stats.csv", index=False)
    
    # Simple Plot
    plt.figure(figsize=(10,6))
    timestamps = df['DateTime'].to_pandas().values
    
    # Plot top 3
    top_idxs = np.argsort(res_pnl)[-3:]
    for idx in top_idxs:
        w_idx = (idx // num_thresh) // num_mults
        t_idx = idx % num_thresh
        w = windows[w_idx]
        vol = min_atrs[t_idx]
        lbl = f"W={w} Vol={vol} PnL={res_pnl[idx]:.3f}"
        plt.plot(timestamps, res_eq[:, idx].get() if hasattr(res_eq, 'get') else res_eq[:, idx], label=lbl) # Fix for get() in loop if needed
        
    plt.legend()
    plt.title("Top 3 Strategies (All Hours + Vol Filter)")
    plt.savefig(f"{OUTPUT_DIR}/equity.png")
    print("Saved plots.")

if __name__ == "__main__":
    run_strategy()
