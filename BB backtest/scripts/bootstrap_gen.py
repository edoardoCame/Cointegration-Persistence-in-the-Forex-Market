import cudf
import numpy as np
import cupy as cp
from tqdm import tqdm
import pandas as pd
import os

def generate_weekly_bootstrap(input_path, output_path, n_paths=100):
    """Generatore di path con shuffle settimanale integrato"""
    print(f"Loading data from {input_path}...")
    df = cudf.read_csv(input_path, sep=";", names=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = cudf.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
    
    pdf_dates = df['timestamp'].to_pandas()
    df['week_id'] = (pdf_dates.dt.year.astype(str) + "_" + pdf_dates.dt.isocalendar().week.astype(str)).values
    weeks = df['week_id'].unique().to_arrow().to_pylist()
    
    week_groups = [df[df['week_id'] == w][['timestamp', 'close']].reset_index(drop=True) for w in weeks if len(df[df['week_id'] == w]) > 0]
    n_weeks, target_len = len(week_groups), len(df)
    
    print(f"Generating {n_paths} paths via weekly shuffle...")
    all_paths = {'timestamp': df['timestamp'].to_pandas()}
    
    for i in tqdm(range(n_paths), desc="Bootstrapping"):
        perm = cp.random.permutation(n_weeks).get()
        path_returns = [cp.diff(cp.log(week_groups[idx]['close'].values)) for idx in perm]
        all_ret_vec = cp.concatenate(path_returns)
        initial_price = week_groups[perm[0]]['close'].iloc[0]
        reconstructed = initial_price * cp.exp(cp.concatenate([cp.array([0.0]), cp.cumsum(all_ret_vec)]))
        
        if len(reconstructed) > target_len: res = reconstructed[:target_len].get()
        else: res = cp.pad(reconstructed, (0, target_len - len(reconstructed)), mode='edge').get()
        all_paths[f'path_{i}'] = res

    print(f"Saving to {output_path}...")
    pd.DataFrame(all_paths).to_parquet(output_path)
    print(f"Done.")

if __name__ == "__main__":
    input_csv = "data/DAT_ASCII_EURGBP_M1_2025.csv"
    output_pq = "data/bootstrap_paths.parquet"
    
    try:
        user_input = input("Enter number of paths to generate [default 100]: ").strip()
        n_paths = int(user_input) if user_input else 100
    except ValueError:
        print("Invalid input. Using default: 100")
        n_paths = 100

    if os.path.exists(output_pq):
        print(f"Warning: {output_pq} already exists and will be overwritten.")
    
    generate_weekly_bootstrap(input_csv, output_pq, n_paths)
