
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
from scipy import stats

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

def load_data_cudf(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "DAT_ASCII_GBPJPY_M1_20*.csv")))
    print(f"Found {len(files)} files.")
    dfs = []
    for f in files:
        try:
            df = cudf.read_csv(f, sep=';', names=['date_str', 'open', 'high', 'low', 'close', 'volume'], header=None)
            df['time'] = cudf.to_datetime(df['date_str'].astype(str), format='%Y%m%d %H%M%S')
            df = df.drop(columns=['date_str', 'open', 'high', 'low', 'volume']) 
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        raise ValueError("No data loaded")
        
    full_df = cudf.concat(dfs).sort_values('time').reset_index(drop=True)
    return full_df

def calculate_rsi_matrix_gpu(close_array_gpu, lookbacks):
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
    
    print(f"Calculating RSI on GPU for {n_lb} lookbacks...")
    t0 = time.time()
    
    rsi_kernel_cuda[blocks, threads_per_block](gain, loss, lookbacks_gpu, out_rsi)
    cuda.synchronize()
    
    print(f"RSI GPU Calc finished in {time.time()-t0:.4f}s")
    
    return out_rsi

def simulate_strategy_returns(rsi_arr, prices_cpu, price_diff_cpu, upper, lower, cost):
    """
    Fast CPU-based strategy simulation for one set of params.
    Returns: pct_returns array
    """
    # Logic
    is_long = rsi_arr > upper
    is_short = rsi_arr < lower
    
    events = np.zeros_like(rsi_arr, dtype=np.int8)
    events[is_long] = 1
    events[is_short] = 2
    
    # FFill
    idx = np.arange(len(events))
    mask = events != 0
    valid_idx = np.where(mask, idx, 0)
    last_event_idx = np.maximum.accumulate(valid_idx)
    
    final_events = events[last_event_idx]
    pos = np.where(final_events == 2, -1, final_events)
    
    # Cost
    pos_prev = np.roll(pos, 1)
    pos_prev[0] = 0
    
    trade_cost = np.abs(pos - pos_prev) * (cost / 2.0)
    
    # Returns
    gross_ret = pos_prev * price_diff_cpu
    net_ret = gross_ret - trade_cost
    
    # Pct Return
    prices_prev = np.roll(prices_cpu, 1)
    prices_prev[0] = prices_cpu[0]
    
    pct_ret = net_ret / prices_prev
    
    return pct_ret

def run_monte_carlo():
    # --------------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------------
    POSSIBLE_LOOKBACKS = np.arange(60, 30001, 60)
    POSSIBLE_UPPER = np.arange(55, 96, 5) 
    POSSIBLE_LOWER = np.arange(5, 46, 5)
    
    N_STRATEGIES = 10
    N_SIMULATIONS = 1000
    COST_PER_FLIP = 0.012 # 1.2 pips (JPY pairs)
    INITIAL_CAPITAL = 10000.0
    
    # PATHS
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results/monte_carlo")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------------------------
    print("="*60)
    print("MONTE CARLO PORTFOLIO SIMULATION - 1000 RUNS")
    print("="*60)
    
    t0 = time.time()
    full_df = load_data_cudf(DATA_DIR)
    
    prices_gpu = cp.asarray(full_df['close'].values, dtype=cp.float32)
    times_pd = full_df['time'].to_pandas()
    
    print(f"Data Loaded: {len(full_df)} rows.")
    
    # --------------------------------------------------------------------------
    # 2. PRE-CALCULATE ALL RSIs (Full Range)
    # --------------------------------------------------------------------------
    print("\nPre-calculating ALL RSI lookbacks for maximum parallelism...")
    # We'll calculate ALL possible lookbacks (500 lookbacks)
    # Memory: 500 x 371k x 4 bytes = 0.74 GB. OK.
    
    all_lookbacks = POSSIBLE_LOOKBACKS.tolist()
    rsi_matrix_gpu = calculate_rsi_matrix_gpu(prices_gpu, all_lookbacks)
    
    # Map lookback value to row
    lb_to_idx = {lb: i for i, lb in enumerate(all_lookbacks)}
    
    # Transfer to CPU for fast indexing (OR keep on GPU and transfer slices)
    # CPU is easier for indexing, and 0.74GB transfer is ~100ms
    print("Transferring RSI matrix to CPU...")
    rsi_matrix_cpu = cp.asnumpy(rsi_matrix_gpu)
    
    # Also prepare price data on CPU
    prices_cpu = cp.asnumpy(prices_gpu)
    price_diff = np.diff(prices_cpu)
    price_diff_padded = np.concatenate(([0.0], price_diff))
    
    print(f"Pre-calculation complete in {time.time()-t0:.2f}s")
    
    # --------------------------------------------------------------------------
    # 3. GENERATE ALL RANDOM STRATEGY SETS (1000 x 10)
    # --------------------------------------------------------------------------
    print(f"\nGenerating {N_SIMULATIONS} random strategy sets...")
    
    all_sim_strategies = []
    
    for sim_idx in range(N_SIMULATIONS):
        strategies = []
        seen = set()
        
        while len(strategies) < N_STRATEGIES:
            lb = int(np.random.choice(POSSIBLE_LOOKBACKS))
            up = int(np.random.choice(POSSIBLE_UPPER))
            low = int(np.random.choice(POSSIBLE_LOWER))
            
            cfg = (lb, up, low)
            if cfg not in seen:
                seen.add(cfg)
                strategies.append(cfg)
                
        all_sim_strategies.append(strategies)
        
    print(f"Generated {N_SIMULATIONS} sets.")
    
    # --------------------------------------------------------------------------
    # 4. RUN SIMULATIONS (Vectorized where possible)
    # --------------------------------------------------------------------------
    print(f"\nRunning {N_SIMULATIONS} portfolio simulations...")
    
    final_equities = []
    
    t_sim_start = time.time()
    
    for sim_idx, strategies in enumerate(all_sim_strategies):
        if (sim_idx + 1) % 100 == 0:
            elapsed = time.time() - t_sim_start
            rate = (sim_idx + 1) / elapsed
            eta = (N_SIMULATIONS - sim_idx - 1) / rate
            print(f"  Sim {sim_idx+1}/{N_SIMULATIONS} | Rate: {rate:.1f} sim/s | ETA: {eta:.1f}s")
        
        # Calculate returns for this sim's 10 strategies
        all_strat_returns = pd.DataFrame(index=times_pd)
        
        for i, (lb, up, low) in enumerate(strategies):
            row_idx = lb_to_idx[lb]
            rsi_arr = rsi_matrix_cpu[row_idx, :]
            
            pct_ret = simulate_strategy_returns(
                rsi_arr, prices_cpu, price_diff_padded, up, low, COST_PER_FLIP
            )
            
            all_strat_returns[f'strat_{i}'] = pct_ret
            
        # Weekly Rebalancing
        all_strat_returns['week'] = all_strat_returns.index.to_period('W')
        grouped = all_strat_returns.groupby('week')
        
        week_keys = sorted(list(grouped.groups.keys()))
        
        current_capital = INITIAL_CAPITAL
        
        for w in week_keys:
            week_data = grouped.get_group(w).drop(columns='week')
            if week_data.empty: 
                continue
            
            week_growth = (1 + week_data).cumprod()
            alloc_per_strat = current_capital / N_STRATEGIES
            week_value_series = (week_growth * alloc_per_strat).sum(axis=1)
            
            current_capital = week_value_series.iloc[-1]
            
        final_equities.append(current_capital)
        
    print(f"\nAll simulations complete in {time.time() - t_sim_start:.2f}s")
    
    # --------------------------------------------------------------------------
    # 5. ANALYSIS & PLOTTING
    # --------------------------------------------------------------------------
    final_equities = np.array(final_equities)
    
    print("\n" + "="*60)
    print("MONTE CARLO RESULTS")
    print("="*60)
    print(f"Mean Final Equity:   {final_equities.mean():.2f}")
    print(f"Median Final Equity: {np.median(final_equities):.2f}")
    print(f"Std Dev:             {final_equities.std():.2f}")
    print(f"Min:                 {final_equities.min():.2f}")
    print(f"Max:                 {final_equities.max():.2f}")
    print(f"Profitable Runs:     {(final_equities > INITIAL_CAPITAL).sum()} / {N_SIMULATIONS} ({100*(final_equities > INITIAL_CAPITAL).sum()/N_SIMULATIONS:.1f}%)")
    print("="*60)
    
    # Save Results
    results_df = pd.DataFrame({'final_equity': final_equities})
    results_df['return_pct'] = (results_df['final_equity'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    results_df.to_csv(os.path.join(RESULTS_DIR, "monte_carlo_results.csv"), index=False)
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Histogram with KDE
    ax1 = axes[0]
    ax1.hist(final_equities, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    
    # Fit normal distribution
    mu, sigma = final_equities.mean(), final_equities.std()
    x = np.linspace(final_equities.min(), final_equities.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal Fit (μ={mu:.0f}, σ={sigma:.0f})')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(final_equities)
    ax1.plot(x, kde(x), 'g-', linewidth=2, label='KDE')
    
    ax1.axvline(INITIAL_CAPITAL, color='black', linestyle='--', linewidth=2, label='Initial Capital')
    ax1.axvline(mu, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean')
    
    ax1.set_xlabel('Final Equity ($)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution of Final Equity - {N_SIMULATIONS} Monte Carlo Runs\nRSI Breakout Portfolio (GBPJPY, N={N_STRATEGIES}, Weekly Rebal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Return % Distribution
    ax2 = axes[1]
    returns_pct = (final_equities - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    ax2.hist(returns_pct, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Breakeven')
    ax2.axvline(returns_pct.mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {returns_pct.mean():.1f}%')
    
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "monte_carlo_distribution.png"), dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Total Runtime: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    run_monte_carlo()
