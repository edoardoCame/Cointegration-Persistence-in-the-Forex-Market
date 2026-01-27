
import sys
import os
import time
import math
import numpy as np
import cudf
import cupy as cp
from numba import cuda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import utils

OUTPUT_DIR = "./donchian breakout/backtest_eurusd/rsi/results"
COST_TOTAL = 0.00008

@cuda.jit
def rsi_kernel_batch(opens, closes, rsis,
                     thresholds_high, thresholds_low, hold_periods,
                     res_pnl, res_trades, res_equity,
                     save_equity):
    """
    Batch kernel for RSI.
    """
    idx = cuda.grid(1)
    if idx >= thresholds_high.shape[0]: return # Limit
    
    thresh_h = thresholds_high[idx]
    thresh_l = thresholds_low[idx]
    max_hold = hold_periods[idx]
    
    n_bars = len(closes)
    position = 0
    entry_price = 0.0
    entry_idx = 0
    equity = 0.0
    trade_count = 0
    
    # RSI is precomputed per-period on host/gpu before calling?
    # Actually, if we test multiple RSI periods, we need multiple RSI arrays.
    # The batching logic assumes ONE indicator array usually.
    # Solution: We must pass a MATRIX of RSIs if we vary Period.
    # Or simplified: Run Batch 1 for Period 14, Batch 2 for Period 60...
    # Here `rsis` is 1D. This means this kernel runs for ONE RSI period.
    # The Outer Loop must handle RSI Period iterations.
    
    for i in range(1, n_bars-1):
        curr_rsi = rsis[i]
        prev_rsi = rsis[i-1]
        
        # Exits
        if position != 0:
            if (i - entry_idx) >= max_hold:
                 pnl = 0.0
                 if position == 1: pnl = (opens[i+1] - entry_price) - COST_TOTAL
                 else: pnl = (entry_price - opens[i+1]) - COST_TOTAL
                 equity += pnl
                 position = 0
                 trade_count += 1
        
        if save_equity:
             res_equity[i, idx] = equity # Simplified mark to market
             
        # Entries
        if position == 0 and (not math.isnan(curr_rsi)) and (not math.isnan(prev_rsi)):
            if prev_rsi < thresh_h and curr_rsi >= thresh_h:
                position = 1
                entry_price = opens[i+1]
                entry_idx = i+1
            elif prev_rsi > thresh_l and curr_rsi <= thresh_l:
                position = -1
                entry_price = opens[i+1]
                entry_idx = i+1
    
    res_pnl[idx] = equity
    res_trades[idx] = trade_count

def calculate_rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def run_large_scale_rsi():
    df = utils.load_data()
    
    d_opens = cp.asarray(df['Open'].values)
    d_closes = cp.asarray(df['Close'].values)
    d_close_series = df['Close'] # Keep reference for calculation
    n_bars = len(df)
    
    # --- MASSIVE PARAMS ---
    # Varying Period requires re-calculating RSI.
    # Periods: M15, H1, H4, D1 approx equivalent
    p_periods = [14, 28, 60, 240, 480, 1440] 
    
    # Thresholds: (High, Low)
    p_thresh = [
        (55, 45), (60, 40), (65, 35), (70, 30), (75, 25), (80, 20)
    ]
    
    # Holds:
    p_holds = [60, 240, 960, 1440, 2880, 7200, 14400]
    
    all_results = []
    
    # OUTER LOOP: Periods (Since we need to regen input data)
    for period in p_periods:
        print(f"Index RSI Period {period}...")
        rsi_vals = calculate_rsi(d_close_series, period).values
        d_rsis = cp.asarray(rsi_vals)
        
        # Inner Combos for this period
        inner_combos = []
        for t in p_thresh:
            for h in p_holds:
                inner_combos.append((t[0], t[1], h))
                
        # Batching Inner Combos
        BATCH_SIZE = 200
        for i in range(0, len(inner_combos), BATCH_SIZE):
            batch = inner_combos[i : i+BATCH_SIZE]
            curr_size = len(batch)
            
            b_th = cp.asarray([x[0] for x in batch], dtype=cp.float32)
            b_tl = cp.asarray([x[1] for x in batch], dtype=cp.float32)
            b_hol = cp.asarray([x[2] for x in batch], dtype=cp.int32)
            
            d_pnl = cp.zeros(curr_size, dtype=cp.float64)
            d_trd = cp.zeros(curr_size, dtype=cp.int32)
            d_eq_dummy = cp.zeros((1,1), dtype=cp.float64)
            
            threads = 128
            blocks = (curr_size + threads - 1) // threads
            
            rsi_kernel_batch[blocks, threads](
                d_opens, d_closes, d_rsis,
                b_th, b_tl, b_hol,
                d_pnl, d_trd, d_eq_dummy, False
            )
            cuda.synchronize()
            
            h_pnl = d_pnl.get()
            h_trd = d_trd.get()
            
            for k in range(curr_size):
                all_results.append({
                    "Label": f"RSI{period}_H{batch[k][0]}_L{batch[k][1]}_Hold{batch[k][2]}",
                    "Period": period,
                    "TH": batch[k][0],
                    "TL": batch[k][1],
                    "Hold": batch[k][2],
                    "PnL": h_pnl[k],
                    "Trades": h_trd[k]
                })

    # --- ANALYZE ---
    import pandas as pd
    res_df = pd.DataFrame(all_results)
    utils.save_stats(res_df, OUTPUT_DIR)
    
    top_5 = res_df.sort_values("PnL", ascending=False).head(5)
    print("Top 5 RSI Strategies:")
    print(top_5[['Label', 'PnL', 'Trades']])
    
    # Rerun Top 5
    # Be careful: Top 5 might have DIFFERENT RSI periods.
    # We cannot run them in one kernel batch if input RSI array implies one period.
    # We must loop through Top 5 one by one to regenerate correct RSI.
    
    print("Re-running Top 5 (Individually due to Period variance)...")
    
    equity_curves = []
    labels = []
    
    # Manual loop
    combined_equity = np.zeros((n_bars, 5), dtype=np.float64) # Host array
    
    idx_col = 0
    for idx, row in top_5.iterrows():
        period = int(row['Period'])
        th = float(row['TH'])
        tl = float(row['TL'])
        hold = int(row['Hold'])
        
        # Re-calc RSI for this specific winner
        rsi_vals = calculate_rsi(d_close_series, period).values
        d_rsis = cp.asarray(rsi_vals)
        
        # Single launch
        b_th = cp.asarray([th], dtype=cp.float32)
        b_tl = cp.asarray([tl], dtype=cp.float32)
        b_hol = cp.asarray([hold], dtype=cp.int32)
        d_pnl = cp.zeros(1, dtype=cp.float64)
        d_trd = cp.zeros(1, dtype=cp.int32)
        d_eq = cp.zeros((n_bars, 1), dtype=cp.float64)
        
        rsi_kernel_batch[1, 128](
            d_opens, d_closes, d_rsis,
            b_th, b_tl, b_hol,
            d_pnl, d_trd, d_eq, True
        )
        cuda.synchronize()
        
        # Copy to host
        eq_host = d_eq.get()[:, 0]
        combined_equity[:, idx_col] = eq_host
        labels.append(f"RSI{period}_{th}/{tl}_H{hold} (PnL={row['PnL']:.2f})")
        idx_col += 1
        
    # SAVE CSV for Portfolio
    import pandas as pd
    timestamps = df['DateTime'].to_pandas()
    df_eq = pd.DataFrame(combined_equity, columns=labels, index=timestamps)
    csv_path = os.path.join(OUTPUT_DIR, "top_5_equities.csv")
    df_eq.to_csv(csv_path)
    print(f"Saved top 5 equities to {csv_path}")

    # Convert numpy matrix back to something utils likes (can pass numpy)
    # Utils expects cupy usually but we handled pandas. Let's pass Cupy subset
    d_combined_subset = cp.asarray(combined_equity)
    utils.plot_equity(df['DateTime'], d_combined_subset, labels, OUTPUT_DIR)

if __name__ == "__main__":
    run_large_scale_rsi()
