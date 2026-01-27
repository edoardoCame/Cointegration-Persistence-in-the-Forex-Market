
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

OUTPUT_DIR = "./donchian breakout/backtest_eurusd/vol_breakout/results"
COST_TOTAL = 0.00008

@cuda.jit
def vol_kernel_batch(opens, highs, lows, closes, atrs,
                     multipliers, hold_periods,
                     res_pnl, res_trades, res_equity,
                     save_equity):
    """
    Batch Kernel for Volatility Breakout.
    """
    idx = cuda.grid(1)
    
    if idx >= multipliers.shape[0]: return # Limit by batch
    
    mult = multipliers[idx]
    max_hold = hold_periods[idx]
    
    n_bars = len(closes)
    position = 0
    entry_price = 0.0
    entry_idx = 0
    equity = 0.0
    trade_count = 0
    
    for i in range(1, n_bars-1):
        curr_close = closes[i]
        curr_open = opens[i]
        curr_atr = atrs[i]
        
        if position != 0:
            # Simple Time Exit
            bars_held = i - entry_idx
            if bars_held >= max_hold:
                 pnl = 0.0
                 # Exit at Open[i+1] ideally, but logic here checks at [i]
                 exit_p = opens[i] 
                 if position == 1: pnl = (opens[i+1] - entry_price) - COST_TOTAL
                 else: pnl = (entry_price - opens[i+1]) - COST_TOTAL
                 equity += pnl
                 position = 0
                 trade_count += 1
        
        # Save Equity?
        if save_equity:
            unrealized = 0.0
            if position == 1: unrealized = (curr_close - entry_price) - COST_TOTAL/2
            elif position == -1: unrealized = (entry_price - curr_close) - COST_TOTAL/2
            res_equity[i, idx] = equity + unrealized
        
        # Entry Logic
        if position == 0 and (not math.isnan(curr_atr)):
            threshold = curr_atr * mult
            if (curr_close - curr_open) > threshold:
                position = 1
                entry_price = opens[i+1]
                entry_idx = i+1
            elif (curr_open - curr_close) > threshold:
                position = -1
                entry_price = opens[i+1]
                entry_idx = i+1
    
    res_pnl[idx] = equity
    res_trades[idx] = trade_count

def run_large_scale_vol():
    df = utils.load_data()
    
    # Pre-calc ATR (Host -> GPU)
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = (high - low)
    tr = tr1 # Simple Range
    # standard ATR calculation too slow? no, TR helps standardizing.
    # Recalculating full ATR here
    tr2 = (high - prev_close).abs()
    tr3 = (prev_close - low).abs()
    tr_final = cudf.DataFrame({'a':tr1,'b':tr2,'c':tr3}).max(axis=1)
    atr = tr_final.rolling(14).mean().fillna(0.0)
    
    d_atrs = cp.asarray(atr.values)
    d_opens = cp.asarray(df['Open'].values)
    d_closes = cp.asarray(df['Close'].values)
    d_highs = cp.asarray(df['High'].values)
    d_lows = cp.asarray(df['Low'].values)
    n_bars = len(df)
    
    # --- MASSIVE PARAMETERS ---
    # Mults: 2.0 to 10.0 step 0.5
    p_mults = np.arange(2.0, 10.5, 0.5)
    # Holds: 4h to 14 days
    p_holds = [240, 480, 960, 1440, 2880, 4320, 7200, 10080, 14400, 20160]
    
    all_combos = []
    for m in p_mults:
        for h in p_holds:
            all_combos.append((m, h))
            
    total_strategies = len(all_combos)
    print(f"Total Vol Strategies: {total_strategies}")
    
    # --- BATCH ---
    BATCH_SIZE = 200
    all_results = []
    
    for i in range(0, total_strategies, BATCH_SIZE):
        batch = all_combos[i : i+BATCH_SIZE]
        curr_batch_size = len(batch)
        print(f"Batch {i}...")
        
        b_mults = cp.asarray([x[0] for x in batch], dtype=cp.float32)
        b_holds = cp.asarray([x[1] for x in batch], dtype=cp.int32)
        
        d_pnl = cp.zeros(curr_batch_size, dtype=cp.float64)
        d_trd = cp.zeros(curr_batch_size, dtype=cp.int32)
        d_eq_dummy = cp.zeros((1,1), dtype=cp.float64)
        
        threads = 128
        blocks = (curr_batch_size + threads - 1) // threads
        
        vol_kernel_batch[blocks, threads](
            d_opens, d_highs, d_lows, d_closes, d_atrs,
            b_mults, b_holds,
            d_pnl, d_trd, d_eq_dummy, False
        )
        cuda.synchronize()
        
        h_pnl = d_pnl.get()
        h_trd = d_trd.get()
        
        for k in range(curr_batch_size):
            all_results.append({
                "Label": f"Vol_K{batch[k][0]:.1f}_Hold{batch[k][1]}",
                "Mult": batch[k][0],
                "Hold": batch[k][1],
                "PnL": h_pnl[k],
                "Trades": h_trd[k]
            })
            
    # --- ANALYZE ---
    import pandas as pd
    res_df = pd.DataFrame(all_results)
    utils.save_stats(res_df, OUTPUT_DIR)
    
    top_5 = res_df.sort_values("PnL", ascending=False).head(5)
    print("Top 5 Vol Strategies:")
    print(top_5[['Label', 'PnL', 'Trades']])
    
    # Rerun Top 5
    top_combos = []
    labels = []
    for idx, row in top_5.iterrows():
        top_combos.append((float(row['Mult']), int(row['Hold'])))
        labels.append(f"Vol_K{row['Mult']:.1f}_Hold{int(row['Hold'])} (PnL={row['PnL']:.2f})")
    
    n_top = len(top_combos)
    b_mults = cp.asarray([x[0] for x in top_combos], dtype=cp.float32)
    b_holds = cp.asarray([x[1] for x in top_combos], dtype=cp.int32)
    
    d_pnl = cp.zeros(n_top, dtype=cp.float64)
    d_trd = cp.zeros(n_top, dtype=cp.int32)
    d_equity = cp.zeros((n_bars, n_top), dtype=cp.float64)
    
    threads = 128
    blocks = (n_top + threads - 1) // threads
    
    vol_kernel_batch[blocks, threads](
        d_opens, d_highs, d_lows, d_closes, d_atrs,
        b_mults, b_holds,
        d_pnl, d_trd, d_equity, True
    )
    cuda.synchronize()
    
    # SAVE CSV for Portfolio
    import pandas as pd
    eq_host = d_equity.get()
    
    # Fix Zero Drop
    eq_host[-1, :] = eq_host[-2, :]
    
    timestamps = df['DateTime'].to_pandas()
    
    # Portfolio Export
    port_labels = [f"VOL_{i}" for i in range(len(labels))]
    df_eq_port = pd.DataFrame(eq_host, columns=port_labels, index=timestamps)
    
    port_dir = "/mnt/ssd2/DARWINEX_Mission/donchian breakout/backtest_eurusd/portfolio/components"
    if not os.path.exists(port_dir): os.makedirs(port_dir)
    port_path = os.path.join(port_dir, "vol.csv")
    df_eq_port.to_csv(port_path)
    print(f"Saved Portfolio Components to {port_path}")
    
    # Detailed Export
    df_eq = pd.DataFrame(eq_host, columns=labels, index=timestamps)
    df_eq.to_csv(os.path.join(OUTPUT_DIR, "top_5_equities.csv"))
    print(f"Saved top 5 equities to {os.path.join(OUTPUT_DIR, 'top_5_equities.csv')}")
    
    utils.plot_equity(df['DateTime'], d_equity, labels, OUTPUT_DIR)

if __name__ == "__main__":
    run_large_scale_vol()
