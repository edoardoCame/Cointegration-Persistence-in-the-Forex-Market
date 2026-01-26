import cudf
import cupy as cp
import numpy as np
import pandas as pd
import numba
from numba import jit
from pathlib import Path
from datetime import datetime, timedelta
import logging
import warnings
from tqdm import tqdm

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pairs_trading/backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/mnt/ssd2/DARWINEX_Mission/data")
START_DATE = pd.Timestamp("2025-12-01")
END_DATE = pd.Timestamp("2025-12-31")
LOOKBACK_DAYS = 90
TOP_N_PAIRS = 5
Z_ENTRY = 1.0
Z_EXIT = 0.0 # Mean reversion
CAPITAL_PER_PAIR = 20000.0 # Total 100k split into 5

# --- GPU Cointegration Logic (Adapted) ---

def load_all_data_gpu(data_dir: Path) -> cudf.DataFrame:
    files = sorted(list(data_dir.glob("DAT_ASCII_*.csv")))
    if not files: raise ValueError("No data files")
    frames = []
    logger.info(f"Reading {len(files)} files...")
    for f in tqdm(files, desc="Loading Data"):
        try:
            pair_name = f.name.split('_')[2].upper()
            # Reading only necessary columns
            df = cudf.read_csv(f, sep=';', header=None, usecols=[0, 4], names=['dt', 'close'], dtype={'dt':'str', 'close':'float32'})
            df['timestamp'] = cudf.to_datetime(df['dt'], format='%Y%m%d %H%M%S')
            df = df.drop_duplicates('timestamp').drop(columns='dt').rename(columns={'close': pair_name}).set_index('timestamp')
            frames.append(df)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
            continue
    
    # Merge all
    df_all = cudf.concat(frames, axis=1).sort_index().ffill().bfill()
    return df_all

def get_ols_params(data_cp: cp.ndarray):
    """Compute OLS beta/alpha vectorized."""
    T, N = data_cp.shape
    means = data_cp.mean(axis=0)
    centered = data_cp - means
    cov = cp.dot(centered.T, centered) / (T - 1)
    
    # Variance of X (denominator) is on the diagonal
    var_x = cp.diag(cov)
    
    # Beta_ij = Cov(i, j) / Var(j)
    # Broadcast var_x across rows (so each col j is divided by var_j)
    beta = cov / (var_x[None, :] + 1e-10)
    
    cp.fill_diagonal(beta, 1.0)
    alpha = means[:, None] - beta * means[None, :]
    return beta, alpha

def compute_adf_score(data_cp: cp.ndarray, beta: cp.ndarray, alpha: cp.ndarray):
    """Compute ADF t-stat for spread."""
    T, N = data_cp.shape
    t_stats = cp.zeros((N, N), dtype=cp.float32)
    
    # We can parallelize the loop somewhat or just loop N times (N is small ~28)
    for j in range(N):
        x_j = data_cp[:, j]
        # Spread = Y - (Beta*X + Alpha)
        y_preds = x_j[:, None] * beta[:, j][None, :] + alpha[:, j][None, :]
        spread = data_cp - y_preds
        
        spread_mean = spread.mean(axis=0)
        spread_centered = spread - spread_mean
        
        res = spread_centered
        d_res = res[1:] - res[:-1]
        res_prev = res[:-1]
        
        dot_num = cp.sum(d_res * res_prev, axis=0)
        dot_den = cp.sum(res_prev**2, axis=0)
        # Avoid div zero
        gamma = dot_num / (dot_den + 1e-10)
        
        adf_res = d_res - res_prev * gamma[None, :]
        rss = cp.sum(adf_res**2, axis=0)
        
        se = cp.sqrt((rss / (T - 2)) / (dot_den + 1e-10))
        t_stats[:, j] = gamma / (se + 1e-10)
        
    return t_stats

# --- Numba Trading Logic ---

@jit(nopython=True)
def backtest_intraday(prices_y, prices_x, beta, mean, std, capital, trade_buffer):
    """
    Simulates intraday trading.
    trade_buffer: array of shape (MAX_TRADES, 4) -> [direction, entry_idx, exit_idx, pnl]
    Returns: pnl_accum, num_trades
    """
    n = len(prices_y)
    
    # State
    position = 0 # 0, 1 (Long Spread), -1 (Short Spread)
    
    # Holdings
    qty_y = 0.0
    qty_x = 0.0
    entry_cash_balance = 0.0
    
    # Trade Tracking
    entry_idx = -1
    
    pnl_accum = 0.0
    trades_count = 0
    max_trades_log = trade_buffer.shape[0]
    
    # Pending Orders (for next bar execution)
    next_action = 0 # 0: None, 1: Enter Long, -1: Enter Short, 2: Close
    
    # Iterate
    for i in range(n):
        price_y = prices_y[i]
        price_x = prices_x[i]
        
        # 1. Execute Pending Orders from previous bar
        if next_action != 0:
            if next_action == 2: # Close
                # Close current position
                cash_flow = 0.0
                if position == 1: 
                    # Closing Long Spread: Sell Y (+), Buy X (-)
                    cash_flow = (qty_y * price_y) - (qty_x * price_x)
                elif position == -1:
                    # Closing Short Spread: Buy Y (-), Sell X (+)
                    cash_flow = -(qty_y * price_y) + (qty_x * price_x)
                
                trade_pnl = entry_cash_balance + cash_flow
                pnl_accum += trade_pnl
                
                # Log Trade
                if trades_count < max_trades_log:
                    trade_buffer[trades_count, 0] = position
                    trade_buffer[trades_count, 1] = entry_idx
                    trade_buffer[trades_count, 2] = i
                    trade_buffer[trades_count, 3] = trade_pnl
                    trades_count += 1
                
                # Reset
                position = 0
                qty_y = 0.0
                qty_x = 0.0
                entry_cash_balance = 0.0
                entry_idx = -1
                
            elif next_action == 1: # Enter Long Spread
                if position == 0:
                    position = 1
                    entry_idx = i
                    
                    target_val = capital / 2.0 
                    qty_y = target_val / price_y
                    qty_x = qty_y * beta
                    
                    # Cash Flow: Buy Y (-), Sell X (+)
                    entry_cash_balance = -(qty_y * price_y) + (qty_x * price_x)
                    
            elif next_action == -1: # Enter Short Spread
                 if position == 0:
                    position = -1
                    entry_idx = i
                    
                    target_val = capital / 2.0
                    qty_y = target_val / price_y
                    qty_x = qty_y * beta
                    
                    # Cash Flow: Sell Y (+), Buy X (-)
                    entry_cash_balance = (qty_y * price_y) - (qty_x * price_x)
            
            # Reset action
            next_action = 0
            
        # 2. Update Signal (if not last bar)
        if i < n - 1:
            spread = price_y - beta * price_x
            z = (spread - mean) / std
            
            # EOD Force Close at last bar (handled by loop range, but let's be explicit)
            if i == n - 2: # Next bar is last bar
                if position != 0:
                    next_action = 2
                continue

            # Logic
            if position == 0:
                if z < -1.0: # Spread too low -> Long Spread
                    next_action = 1
                elif z > 1.0: # Spread too high -> Short Spread
                    next_action = -1
            elif position == 1: # Long Spread
                if z >= 0.0: # Mean reversion
                    next_action = 2
            elif position == -1: # Short Spread
                if z <= 0.0: # Mean reversion
                    next_action = 2

    # If still open at end of loop (shouldn't happen with force close, but safety)
    if position != 0:
        # Mark to market close at last price
        cash_flow = 0.0
        if position == 1:
             cash_flow = (qty_y * prices_y[n-1]) - (qty_x * prices_x[n-1])
        elif position == -1:
             cash_flow = -(qty_y * prices_y[n-1]) + (qty_x * prices_x[n-1])
        
        trade_pnl = entry_cash_balance + cash_flow
        pnl_accum += trade_pnl
        
        if trades_count < max_trades_log:
            trade_buffer[trades_count, 0] = position
            trade_buffer[trades_count, 1] = entry_idx
            trade_buffer[trades_count, 2] = n-1
            trade_buffer[trades_count, 3] = trade_pnl
            trades_count += 1

    return pnl_accum, trades_count

# --- Main Execution ---

def main():
    logger.info("Initializing Pairs Trading Backtest...")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    
    # 1. Load Data to GPU
    df_all = load_all_data_gpu(DATA_DIR)
    pairs = df_all.columns.tolist()
    logger.info(f"Loaded {len(pairs)} pairs.")
    
    # Convert to matrix for fast indexing
    data_matrix = cp.asarray(df_all.values, dtype=cp.float32)
    timestamps = df_all.index.to_pandas()
    
    # Date Indexing
    unique_dates = pd.to_datetime(timestamps).floor('D').unique()
    unique_dates = sorted([d for d in unique_dates if d >= (START_DATE - timedelta(days=LOOKBACK_DAYS + 5))])
    
    # Filter for simulation period
    sim_dates = [d for d in unique_dates if d >= START_DATE and d <= END_DATE]
    logger.info(f"Simulation Days: {len(sim_dates)}")
    
    total_pnl = 0.0
    daily_stats = []
    all_trades_log = []
    daily_betas_log = []
    
    for current_date in tqdm(sim_dates, desc="Backtesting Days"):
        # Define Lookback Window: [T-90, T-1]
        window_end_date = current_date - timedelta(days=1)
        window_start_date = current_date - timedelta(days=LOOKBACK_DAYS)
        
        # Get indices
        # We need efficient lookup. 
        # Using boolean mask on pandas index is fast enough for daily slicing
        mask_window = (timestamps >= window_start_date) & (timestamps <= window_end_date)
        if not np.any(mask_window):
            continue
            
        # GPU Slice for Training
        # We need to find integer indices corresponding to the mask to slice the cp array
        # Doing this via numpy searchsorted is faster
        ts_values = timestamps.values
        idx_start = np.searchsorted(ts_values, window_start_date.to_datetime64())
        idx_end = np.searchsorted(ts_values, current_date.to_datetime64()) # Up to current date (exclusive)
        
        if idx_end <= idx_start: continue
        
        train_data = data_matrix[idx_start:idx_end]
        
        # --- Morning Analysis (GPU) ---
        beta_mat, alpha_mat = get_ols_params(train_data)
        adf_stats = compute_adf_score(train_data, beta_mat, alpha_mat)
        
        # Extract Results to CPU
        h_adf = cp.asnumpy(adf_stats)
        h_beta = cp.asnumpy(beta_mat)
        h_alpha = cp.asnumpy(alpha_mat)
        
        # Find Top Pairs
        # Flatten and sort
        # We only care about off-diagonal
        np.fill_diagonal(h_adf, 9999.0)
        
        # Create list of (score, r, c)
        candidates = []
        rows, cols = h_adf.shape
        for r in range(rows):
            for c in range(cols):
                if r == c: continue
                candidates.append((h_adf[r, c], r, c))
        
        # Sort by most negative ADF score
        candidates.sort(key=lambda x: x[0])
        top_picks = candidates[:TOP_N_PAIRS]
        
        # --- Prepare Trading Params ---
        # For the selected pairs, we need Mean and Std of the spread over the window
        selected_pairs_info = []
        
        for score, r, c in top_picks:
            pair_y = pairs[r]
            pair_x = pairs[c]
            beta = h_beta[r, c]
            alpha = h_alpha[r, c]
            
            # Recalculate Spread Stats on Window (GPU)
            # spread = y - beta*x
            # We can use the train_data slice
            y_vec = train_data[:, r]
            x_vec = train_data[:, c]
            spread_vec = y_vec - beta * x_vec
            
            # Using cupy for fast mean/std
            mean_spread = float(spread_vec.mean())
            std_spread = float(spread_vec.std())
            
            selected_pairs_info.append({
                'y_idx': r, 'x_idx': c,
                'pair_y': pair_y, 'pair_x': pair_x,
                'beta': beta,
                'mean': mean_spread,
                'std': std_spread
            })
            
            # Log Daily Beta
            daily_betas_log.append({
                'date': current_date,
                'pair_y': pair_y,
                'pair_x': pair_x,
                'beta': beta,
                'adf_score': score,
                'spread_mean': mean_spread,
                'spread_std': std_spread
            })
            
        # --- Intraday Trading (CPU Loop) ---
        # Get data for the specific day
        day_mask = (timestamps >= current_date) & (timestamps < (current_date + timedelta(days=1)))
        idx_day_start = np.searchsorted(ts_values, current_date.to_datetime64())
        idx_day_end = np.searchsorted(ts_values, (current_date + timedelta(days=1)).to_datetime64())
        
        if idx_day_end <= idx_day_start:
            logger.warning(f"No data for {current_date}")
            continue
            
        day_data_gpu = data_matrix[idx_day_start:idx_day_end]
        day_data_cpu = cp.asnumpy(day_data_gpu) # Transfer needed columns to CPU for Numba
        
        day_pnl = 0.0
        day_trades = 0
        
        trade_buffer = np.zeros((1000, 4), dtype=np.float64) # Buffer for Numba
        
        for p in selected_pairs_info:
            prices_y = day_data_cpu[:, p['y_idx']]
            prices_x = day_data_cpu[:, p['x_idx']]
            
            # Reset buffer
            trade_buffer[:] = 0.0
            
            pnl, trades_count = backtest_intraday(
                prices_y, prices_x, 
                p['beta'], p['mean'], p['std'], 
                CAPITAL_PER_PAIR,
                trade_buffer
            )
            day_pnl += pnl
            day_trades += trades_count
            
            # Extract trades from buffer
            if trades_count > 0:
                for t_i in range(trades_count):
                    # trade_buffer row: [direction, entry_idx, exit_idx, pnl]
                    direction = trade_buffer[t_i, 0]
                    entry_idx = int(trade_buffer[t_i, 1])
                    exit_idx = int(trade_buffer[t_i, 2])
                    t_pnl = trade_buffer[t_i, 3]
                    
                    # Convert minute indices to approximate times
                    # Ideally we would map back to timestamps, but simple offset is ok for now
                    # We know start of day is current_date
                    entry_time = current_date + timedelta(minutes=entry_idx)
                    exit_time = current_date + timedelta(minutes=exit_idx)
                    
                    all_trades_log.append({
                        'date': current_date,
                        'pair_y': p['pair_y'],
                        'pair_x': p['pair_x'],
                        'direction': 'Long Spread' if direction == 1 else 'Short Spread',
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'beta': p['beta'],
                        'pnl': t_pnl
                    })
            
        total_pnl += day_pnl
        
        daily_stats.append({
            'date': current_date,
            'pnl': day_pnl,
            'trades': day_trades,
            'cum_pnl': total_pnl
        })
        
        logger.info(f"Date: {current_date.date()} | PnL: {day_pnl:.2f} | Trades: {day_trades}")

    # Results
    res_df = pd.DataFrame(daily_stats)
    if not res_df.empty:
        res_df.set_index('date', inplace=True)
        res_df.to_csv("pairs_trading/backtest_results.csv")
        
        # Save detailed logs
        pd.DataFrame(all_trades_log).to_csv("pairs_trading/detailed_trades.csv", index=False)
        pd.DataFrame(daily_betas_log).to_csv("pairs_trading/daily_betas.csv", index=False)
        
        logger.info("\n=== Final Results ===")
        logger.info(f"Total PnL: {total_pnl:.2f}")
        logger.info(f"Total Trades: {res_df['trades'].sum()}")
        logger.info(f"Sharpe (Daily): {res_df['pnl'].mean() / res_df['pnl'].std() * np.sqrt(252):.2f}")
    else:
        logger.info("No trades executed.")

if __name__ == "__main__":
    main()
