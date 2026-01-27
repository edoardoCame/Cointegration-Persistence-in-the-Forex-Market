"""
RSI Portfolio Random Backtest - Multi-Pair Version
Backtests 10 random RSI parameter combinations across ALL currency pairs
with proper pip cost adjustments (1.5 pips round-trip for all pairs).

JPY pairs use 0.01 pip increments, others use 0.0001
"""
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
    alpha = 1.0 / n
    run_alpha = 1.0 - alpha
    
    n_samples = gains.shape[0]
    
    avg_gain = gains[0]
    avg_loss = losses[0]
    
    if avg_loss == 0:
        out_rsi[lb_idx, 0] = 100.0
    else:
        rs = avg_gain / avg_loss
        out_rsi[lb_idx, 0] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(1, n_samples):
        g = gains[i]
        l = losses[i]
        
        avg_gain = run_alpha * avg_gain + alpha * g
        avg_loss = run_alpha * avg_loss + alpha * l
        
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

def load_pair_data(data_dir, pair):
    """Load all CSV files for a single currency pair"""
    files = sorted(glob.glob(os.path.join(data_dir, f"DAT_ASCII_{pair}_M1_*.csv")))
    
    if not files:
        print(f"  No files found for {pair}")
        return None
        
    dfs = []
    for f in files:
        try:
            df = cudf.read_csv(f, sep=';', names=['date_str', 'open', 'high', 'low', 'close', 'volume'], header=None)
            df['time'] = cudf.to_datetime(df['date_str'].astype(str), format='%Y%m%d %H%M%S')
            df = df.drop(columns=['date_str', 'open', 'high', 'low', 'volume'])
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping {f}: {e}")
            
    if not dfs:
        return None
        
    full_df = cudf.concat(dfs).sort_values('time').reset_index(drop=True)
    return full_df

def calculate_rsi_matrix_gpu(close_array_gpu, lookbacks):
    """Calculate RSI for multiple lookbacks using GPU"""
    delta = cp.diff(close_array_gpu)
    delta_padded = cp.concatenate((cp.array([0.0], dtype=cp.float32), delta))
    
    gain = delta_padded.copy()
    loss = delta_padded.copy()
    gain[gain < 0] = 0.0
    loss[loss > 0] = 0.0
    loss = -loss
    
    lookbacks_gpu = cp.array(lookbacks, dtype=cp.int32)
    n_lb = len(lookbacks)
    n_time = len(close_array_gpu)
    
    out_rsi = cp.full((n_lb, n_time), np.nan, dtype=cp.float32)
    
    threads_per_block = 128
    blocks = (n_lb + threads_per_block - 1) // threads_per_block
    
    rsi_kernel_cuda[blocks, threads_per_block](gain, loss, lookbacks_gpu, out_rsi)
    cuda.synchronize()
    
    return out_rsi

def get_cost_per_pip(pair):
    """
    Determine cost per pip based on pair.
    1.5 pips total round-trip cost
    """
    if 'JPY' in pair:
        # JPY pairs: 1 pip = 0.01
        return 0.015  # 1.5 pips
    else:
        # Other pairs: 1 pip = 0.0001
        return 0.00015  # 1.5 pips

def backtest_pair(pair, data_dir, results_dir, num_strategies=10):
    """
    Run random portfolio backtest for a single currency pair.
    Returns: (pair, final_equity, success)
    """
    print(f"\n{'='*70}")
    print(f"Processing: {pair}")
    print(f"{'='*70}")
    
    # Configuration
    POSSIBLE_LOOKBACKS = np.arange(60, 30001, 60)
    POSSIBLE_UPPER = np.arange(55, 96, 5) 
    POSSIBLE_LOWER = np.arange(5, 46, 5)
    
    COST_PER_FLIP = get_cost_per_pip(pair)
    INITIAL_CAPITAL = 10000.0
    
    # Load data
    print(f"Loading data for {pair}...")
    full_df = load_pair_data(data_dir, pair)
    
    if full_df is None or len(full_df) < 1000:
        print(f"  ERROR: Insufficient data for {pair}")
        return (pair, None, False)
    
    print(f"  Data loaded: {len(full_df)} rows")
    
    # Select random strategies
    print(f"Selecting {num_strategies} random strategies...")
    strategies = []
    seen = set()
    
    while len(strategies) < num_strategies:
        lb = int(np.random.choice(POSSIBLE_LOOKBACKS))
        up = int(np.random.choice(POSSIBLE_UPPER))
        low = int(np.random.choice(POSSIBLE_LOWER))
        
        cfg = (lb, up, low)
        if cfg not in seen:
            seen.add(cfg)
            strategies.append(cfg)
    
    # Calculate RSI
    prices_gpu = cp.asarray(full_df['close'].values, dtype=cp.float32)
    times_pd = full_df['time'].to_pandas()
    
    print(f"Calculating RSI...")
    t0 = time.time()
    unique_lbs = sorted(list(set([s[0] for s in strategies])))
    lb_map = {lb: i for i, lb in enumerate(unique_lbs)}
    rsi_matrix = calculate_rsi_matrix_gpu(prices_gpu, unique_lbs)
    print(f"  RSI calculation: {time.time()-t0:.2f}s")
    
    # Simulate strategies
    print(f"Simulating strategies...")
    price_diff = cp.diff(prices_gpu)
    price_diff_padded = cp.concatenate((cp.array([0.0], dtype=cp.float32), price_diff))
    
    all_strat_returns = pd.DataFrame(index=times_pd)
    
    for i, (lb, up, low) in enumerate(strategies):
        row_idx = lb_map[lb]
        rsi_arr = rsi_matrix[row_idx, :]
        
        # CPU-side logic
        rsi_cpu = cp.asnumpy(rsi_arr)
        
        is_long_cpu = rsi_cpu > up
        is_short_cpu = rsi_cpu < low
        
        events_cpu = np.zeros_like(rsi_cpu, dtype=np.int8)
        events_cpu[is_long_cpu] = 1
        events_cpu[is_short_cpu] = 2
        
        idx = np.arange(len(events_cpu))
        mask = events_cpu != 0
        valid_idx = np.where(mask, idx, 0)
        last_event_idx = np.maximum.accumulate(valid_idx)
        
        final_events = events_cpu[last_event_idx]
        pos_cpu = np.where(final_events == 2, -1, final_events)
        
        pos_prev = np.roll(pos_cpu, 1)
        pos_prev[0] = 0
        
        trade_cost = np.abs(pos_cpu - pos_prev) * (COST_PER_FLIP / 2.0)
        
        diff_cpu = cp.asnumpy(price_diff_padded)
        prices_cpu = cp.asnumpy(prices_gpu)
        
        gross_ret = pos_prev * diff_cpu
        net_ret = gross_ret - trade_cost
        
        prices_prev = np.roll(prices_cpu, 1)
        prices_prev[0] = prices_cpu[0]
        
        pct_ret = net_ret / prices_prev
        
        all_strat_returns[f'strat_{i}'] = pct_ret
    
    # Portfolio rebalancing
    print(f"Simulating rebalanced portfolio...")
    all_strat_returns['week'] = all_strat_returns.index.to_period('W')
    grouped = all_strat_returns.groupby('week')
    
    week_keys = sorted(list(grouped.groups.keys()))
    
    port_series_list = []
    current_capital = INITIAL_CAPITAL
    
    for w in week_keys:
        week_data = grouped.get_group(w).drop(columns='week')
        if week_data.empty: 
            continue
        
        week_growth = (1 + week_data).cumprod()
        alloc_per_strat = current_capital / num_strategies
        week_value_series = (week_growth * alloc_per_strat).sum(axis=1)
        
        port_series_list.append(week_value_series)
        current_capital = week_value_series.iloc[-1]
    
    portfolio_equity_curve = pd.concat(port_series_list)
    
    # Save results
    pair_results_dir = os.path.join(results_dir, pair)
    os.makedirs(pair_results_dir, exist_ok=True)
    
    portfolio_equity_curve.to_csv(os.path.join(pair_results_dir, "equity_curve.csv"))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_equity_curve.index, portfolio_equity_curve, linewidth=2, color='steelblue')
    plt.title(f'RSI Portfolio Random - {pair}\nFinal: ${current_capital:.2f} | Cost: {COST_PER_FLIP*100:.3f}%')
    plt.ylabel('Equity ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pair_results_dir, "equity_curve.png"), dpi=100)
    plt.close()
    
    print(f"✓ {pair}: Final Equity = ${current_capital:.2f}")
    
    return (pair, current_capital, True)

def run_all_pairs():
    """Run portfolio backtest for all available pairs"""
    
    # Use absolute paths to avoid issues with conda run
    DATA_DIR = "/mnt/ssd2/DARWINEX_Mission/data"
    RESULTS_DIR = "/mnt/ssd2/DARWINEX_Mission/momentum/backtest/rsi_breakout/results/portfolio"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get all unique pairs
    files = sorted(glob.glob(os.path.join(DATA_DIR, "DAT_ASCII_*.csv")))
    pairs = sorted(list(set([os.path.basename(f).split('_')[2] for f in files])))
    
    print(f"\nFound {len(pairs)} currency pairs")
    print(f"Starting multi-pair RSI portfolio backtest...")
    print(f"Cost: 1.5 pips round-trip")
    
    results_summary = []
    t_total_start = time.time()
    
    for pair in pairs:
        try:
            pair_name, final_eq, success = backtest_pair(pair, DATA_DIR, RESULTS_DIR, num_strategies=10)
            if success:
                results_summary.append({
                    'pair': pair_name,
                    'final_equity': final_eq,
                    'total_return': ((final_eq - 10000) / 10000) * 100,
                    'jpy': 'JPY' in pair_name
                })
        except Exception as e:
            print(f"ERROR on {pair}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - All Pairs Backtested")
    print(f"{'='*70}")
    
    if not results_summary:
        print("ERROR: No results generated")
        return
    
    summary_df = pd.DataFrame(results_summary).sort_values('final_equity', ascending=False)
    
    print(summary_df.to_string(index=False))
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"STATISTICS")
    print(f"{'='*70}")
    print(f"Total Pairs: {len(summary_df)}")
    print(f"Mean Final Equity: ${summary_df['final_equity'].mean():.2f}")
    print(f"Median Final Equity: ${summary_df['final_equity'].median():.2f}")
    print(f"Std Dev: ${summary_df['final_equity'].std():.2f}")
    print(f"Min: ${summary_df['final_equity'].min():.2f}")
    print(f"Max: ${summary_df['final_equity'].max():.2f}")
    print(f"Profitable Pairs: {len(summary_df[summary_df['final_equity'] > 10000])}/{len(summary_df)}")
    
    # Save summary
    summary_df.to_csv(os.path.join(RESULTS_DIR, "all_pairs_summary.csv"), index=False)
    
    # Plot summary bars
    plt.figure(figsize=(14, 8))
    colors = ['green' if x > 10000 else 'red' for x in summary_df['final_equity']]
    plt.barh(summary_df['pair'], summary_df['final_equity'], color=colors)
    plt.axvline(x=10000, color='black', linestyle='--', linewidth=2, label='Initial Capital')
    plt.xlabel('Final Equity ($)')
    plt.title(f'RSI Portfolio Random - All Pairs\n{len(summary_df)} pairs, 10 strategies each, Weekly rebalance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "all_pairs_summary.png"), dpi=100)
    plt.close()
    
    print(f"\nTotal Time: {(time.time()-t_total_start)/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}/")

if __name__ == "__main__":
    run_all_pairs()
