import sys
import os
import cudf
import cupy as cp
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.getcwd())

from returns_research.system import data_feed

def calculate_rolling_theta(series, window=4320):
    """Calculates rolling Theta from OU process using GPU."""
    y = series
    x = series.shift(1)
    
    sum_x = x.rolling(window).sum()
    sum_y = y.rolling(window).sum()
    sum_xx = (x**2).rolling(window).sum()
    sum_xy = (x*y).rolling(window).sum()
    
    N = window
    numerator = N * sum_xy - sum_x * sum_y
    denominator = N * sum_xx - sum_x**2
    
    beta = numerator / denominator
    beta_vals = beta.fillna(np.nan).values
    theta = -cp.log(beta_vals)
    return cudf.Series(theta, index=series.index)

def analyze_predictability(theta_past, theta_future, transform_name, func=None):
    """Calculates decile persistence probabilities."""
    
    # Apply transform if provided
    if func:
        # Filter for valid inputs for the transform (e.g. positive for log/sqrt)
        valid = (theta_past > 0) & (theta_future > 0)
        p = theta_past[valid].copy()
        f = theta_future[valid].copy()
        
        # Apply transform using numpy/cupy logic
        # Moving to pandas for qcut so inputs are numpy/pandas
        p = func(p)
        f = func(f)
    else:
        p = theta_past
        f = theta_future

    # Create DF
    df = pd.DataFrame({'past': p, 'future': f}).dropna()
    
    # Deciles
    df['decile_past'], bins = pd.qcut(df['past'], q=10, retbins=True, labels=False)
    
    # Apply same bins to future?
    # Wait, for invariance test, we should apply qcut independently?
    # NO. The test is "Remain in same Regime".
    # Regime is defined by the distribution of the PAST.
    # So we must use bins from PAST to categorize FUTURE.
    
    # Important: If we transform, the bins transform too.
    # So the logic holds.
    
    df['decile_future'] = pd.cut(df['future'], bins=bins, labels=False, include_lowest=True)
    
    df['stay'] = (df['decile_past'] == df['decile_future']).astype(int)
    
    probs = df.groupby('decile_past')['stay'].mean().values
    return probs, df['past'].corr(df['future']) # Return probs and linear correlation

def main():
    print("Loading EURCHF data...")
    df = data_feed.load_data('EURCHF', timeframe='1min')
    if df is None: return

    WINDOW_PAST = 4320
    WINDOW_FUTURE = 1440
    
    print("Calculating Theta...")
    theta_past_gpu = calculate_rolling_theta(df['Close'], window=WINDOW_PAST)
    theta_future_gpu = calculate_rolling_theta(df['Close'], window=WINDOW_FUTURE).shift(-WINDOW_FUTURE)
    
    # Combine and move to host
    data = cudf.DataFrame({'p': theta_past_gpu, 'f': theta_future_gpu}).dropna()
    # Filter valid theta
    data = data[data['p'] > 0]
    data = data[data['f'] > 0]
    
    p = data['p'].to_pandas()
    f = data['f'].to_pandas()
    
    print(f"\nAnalyzing {len(p)} samples across transformations...\n")
    
    # 1. Raw
    probs_raw, corr_raw = analyze_predictability(p, f, "Raw")
    
    # 2. Sqrt
    probs_sqrt, corr_sqrt = analyze_predictability(p, f, "Sqrt", np.sqrt)
    
    # 3. Log
    probs_log, corr_log = analyze_predictability(p, f, "Log", np.log)
    
    # Comparison
    print(f"{ 'Decile':<8} | { 'Raw Prob':<12} | { 'Sqrt Prob':<12} | { 'Log Prob':<12}")
    print("-" * 50)
    for i in range(10):
        print(f"D{i:<7} | {probs_raw[i]:.2%}       | {probs_sqrt[i]:.2%}       | {probs_log[i]:.2%}")
        
    print("-" * 50)
    print(f"\nLINEAR CORRELATION (Pearson):")
    print(f"Raw:  {corr_raw:.4f}")
    print(f"Sqrt: {corr_sqrt:.4f}")
    print(f"Log:  {corr_log:.4f}")
    
    print("\nCONCLUSION:")
    if np.allclose(probs_raw, probs_sqrt) and np.allclose(probs_raw, probs_log):
        print("Decile Probabilities are IDENTICAL (Invariant to monotonic transform).")
    else:
        print("Decile Probabilities CHANGED (This should not happen if logic is correct).")

if __name__ == "__main__":
    main()
