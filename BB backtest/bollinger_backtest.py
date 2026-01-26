import cudf
import numpy as np
import matplotlib.pyplot as plt
import os

def run_backtest():
    # 1. Load Data
    data_path = "data/DAT_ASCII_EURGBP_M1_2025.csv"
    print(f"Loading data from {data_path}...")
    
    # Format: Timestamp;Open;High;Low;Close;Volume
    df = cudf.read_csv(data_path, sep=";", names=["timestamp", "open", "high", "low", "close", "volume"])
    
    # 2. Parameters
    lookback = 10000
    std_dev_mult = 2.5
    commission_pips = 0.25
    pip_value = 0.0001
    commission_cost = commission_pips * pip_value
    
    # 3. Bollinger Bands
    print("Calculating Bollinger Bands...")
    df['middle_band'] = df['close'].rolling(window=lookback).mean()
    df['std'] = df['close'].rolling(window=lookback).std()
    df['upper_band'] = df['middle_band'] + (df['std'] * std_dev_mult)
    df['lower_band'] = df['middle_band'] - (df['std'] * std_dev_mult)
    
    # Drop rows with NaN from rolling calculations
    df = df.dropna().reset_index(drop=True)
    
    # 4. Strategy Logic (No Lookahead Bias)
    # We generate signals at time T based on data available at time T.
    # We enter/exit at the Close of T (or we can assume Open of T+1, but for M1 data Close(T) is approx Open(T+1))
    # To strictly avoid lookahead, we compare price[t] with bands[t].
    
    close = df['close'].values
    upper = df['upper_band'].values
    lower = df['lower_band'].values
    middle = df['middle_band'].values
    
    # Using a simple loop/logic for position tracking because complex state-dependent signals
    # (like "exit at middle band") are often cleaner to implement with a state machine.
    # However, for GPU performance, we try to vectorize.
    # But since we have "max 1 position", it's a path-dependent process.
    
    # Transitioning to numpy for the path-dependent logic (max 1 position)
    # because cuDF doesn't support complex state-dependent iteration easily without JIT or custom kernels.
    # Since the dataset is 1 year of M1 (approx 525k rows), numpy is fast enough for the logic part.
    
    c_price = close.get()
    u_band = upper.get()
    l_band = lower.get()
    m_band = middle.get()
    
    n = len(c_price)
    positions = np.zeros(n, dtype=np.int8)
    equity = np.zeros(n)
    
    current_pos = 0 # 0: none, 1: long, -1: short
    
    print("Running strategy loop...")
    for i in range(1, n):
        # Default: maintain position
        positions[i] = current_pos
        
        if current_pos == 0:
            # Entry logic
            if c_price[i] < l_band[i]:
                current_pos = 1 # Enter Long
                positions[i] = 1
            elif c_price[i] > u_band[i]:
                current_pos = -1 # Enter Short
                positions[i] = -1
        
        elif current_pos == 1:
            # Exit logic for Long
            if c_price[i] >= m_band[i]:
                current_pos = 0 # Close Long
                positions[i] = 0
        
        elif current_pos == -1:
            # Exit logic for Short
            if c_price[i] <= m_band[i]:
                current_pos = 0 # Close Short
                positions[i] = 0
                
    # 5. Calculate Returns
    # Vectorized return calculation
    df['position'] = positions
    df['price_change'] = df['close'].diff().fillna(0)
    
    # Strategy returns (excluding commissions first)
    # Signal is acted upon at the close of T, so the return of T+1 is position[T] * change[T+1]
    df['strategy_returns'] = df['position'].shift(1).fillna(0) * df['price_change']
    
    # 6. Commissions
    # We pay commission when position changes from 0 to 1/-1 or 1/-1 to 0
    # Also when flipping (though here we close at middle band, so flip is unlikely unless price jumps)
    df['pos_diff'] = df['position'].diff().abs().fillna(0)
    # pos_diff will be 1 if we enter/exit, 2 if we flip (not possible with this logic)
    df['commissions'] = df['pos_diff'] * commission_cost
    
    df['net_returns'] = df['strategy_returns'] - df['commissions']
    df['equity_curve'] = df['net_returns'].cumsum()
    
    # 7. Plotting
    print("Exporting plots...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['equity_curve'].to_pandas())
    plt.title("BB Mean Reversion Equity Curve - EURGBP (2025)")
    plt.xlabel("Minutes")
    plt.ylabel("Cumulative Returns (Price Units)")
    plt.grid(True)
    
    output_dir = "BB backtest"
    plt.savefig(os.path.join(output_dir, "equity_curve.png"))
    
    # Stats
    total_return = df['equity_curve'].iloc[-1]
    print(f"Total Return: {total_return:.6f}")
    
    # Save results to CSV for inspection
    df[['timestamp', 'close', 'upper_band', 'lower_band', 'middle_band', 'position', 'equity_curve']].to_csv(os.path.join(output_dir, "backtest_results.csv"), index=False)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    run_backtest()
