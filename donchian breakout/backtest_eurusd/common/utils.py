
import os
import glob
import cudf
import cupy as cp

DATA_DIR = "/mnt/ssd2/DARWINEX_Mission/data/"

def load_data():
    """
    Loads all EURUSD M1 data from DATA_DIR into a single cuDF DataFrame.
    """
    pattern = os.path.join(DATA_DIR, "DAT_ASCII_EURUSD_M1_*.csv")
    files = sorted(glob.glob(pattern))
    print(f"[Utils] Found {len(files)} data files.")
    
    if not files:
        raise FileNotFoundError(f"No files in {DATA_DIR}")
        
    dfs = []
    for f in files:
        # Assuming format: 20250101 000000;...
        df = cudf.read_csv(f, sep=';', header=None, names=['DateTime','Open','High','Low','Close','Vol'])
        df['DateTime'] = cudf.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
        dfs.append(df)
        
    print("[Utils] Concatenating...")
    df = cudf.concat(dfs, ignore_index=True)
    df = df.sort_values('DateTime').reset_index(drop=True)
    print(f"[Utils] Loaded {len(df)} bars. Range: {df['DateTime'].min()} - {df['DateTime'].max()}")
    return df

def save_stats(results_df, output_dir, filename="stats.csv"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    res_path = os.path.join(output_dir, filename)
    results_df.sort_values("PnL", ascending=False).to_csv(res_path, index=False)
    print(f"[Utils] Saved stats to {res_path}")

def plot_equity(timestamps, equity_curves, labels, output_dir, filename="equity.png", top_n=5):
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(12, 7))
    
    # equity_curves is (Bars, Strategies)
    # Identify top strategies if not already sorted? 
    # Usually passed in active strategies needed.
    # We assume 'equity_curves' contains the data for the 'labels' passed.
    
    # Use pandas for plotting datetime axis easily
    # Fix for cudf NotImpl error on .values for datetime
    if hasattr(timestamps, 'to_pandas'): 
        ts_pd = timestamps.to_pandas()
    elif hasattr(timestamps, 'get'):
        ts_pd = timestamps.get()
        import pandas as pd
        ts_pd = pd.to_datetime(ts_pd)
    else:
        ts_pd = timestamps
        
    for i, lbl in enumerate(labels):
        if i >= top_n: break
        curve = equity_curves[:, i]
        curve_host = curve.get() if hasattr(curve, 'get') else curve
        plt.plot(ts_pd, curve_host, label=lbl) # Plot simplified?
        
    plt.title(f"Top {top_n} Strategies Equity Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    print(f"[Utils] Saved plot to {path}")
