import cudf
import cupy as cp
import pandas as pd
import os

def generate_weekly_bootstrap(input_path, output_path, n_paths=100):
    print(f"Loading data from {input_path}...")
    df = cudf.read_csv(input_path, sep=";", names=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Convert timestamp to datetime and extract week identifiers
    df['timestamp'] = cudf.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
    
    # Use pandas for week grouping as it's more robust for ISO week logic across versions
    pdf_dates = df['timestamp'].to_pandas()
    df['week_id'] = (pdf_dates.dt.year.astype(str) + "_" + pdf_dates.dt.isocalendar().week.astype(str)).values
    
    # Identify unique weeks
    weeks = df['week_id'].unique().to_arrow().to_pylist()
    week_groups = []
    print("Grouping data by weeks...")
    for week in weeks:
        week_df = df[df['week_id'] == week][['timestamp', 'close']].reset_index(drop=True)
        if len(week_df) > 0:
            week_groups.append(week_df)
    
    n_weeks = len(week_groups)
    print(f"Detected {n_weeks} weeks. Generating {n_paths} paths via weekly shuffle...")
    
    all_paths = {}
    # Keep original timestamps for reference
    all_paths['timestamp'] = df['timestamp'].to_pandas()
    
    target_len = len(df)
    
    for i in range(n_paths):
        # Randomly shuffle week order
        perm = cp.random.permutation(n_weeks).get()
        
        path_returns = []
        # We start with the first price of the first week in our random permutation
        initial_price = week_groups[perm[0]]['close'].iloc[0]
        
        for idx in perm:
            week_data = week_groups[idx]
            prices = week_data['close'].values
            # Log returns to avoid gaps and maintain percentage moves
            ret = cp.diff(cp.log(prices))
            path_returns.append(ret)
            
        # Combine all returns
        all_ret_vec = cp.concatenate(path_returns)
        
        # Reconstruct price path: P_t = P_0 * exp(cumsum(returns))
        # Add a zero at the beginning for the initial price
        reconstructed = initial_price * cp.exp(cp.concatenate([cp.array([0.0]), cp.cumsum(all_ret_vec)]))
        
        # Ensure exact length match with original (truncate or pad if necessary due to minute counts)
        if len(reconstructed) > target_len:
            res = reconstructed[:target_len].get()
        else:
            # Pad with last price if slightly shorter
            res = cp.pad(reconstructed, (0, target_len - len(reconstructed)), mode='edge').get()
            
        all_paths[f'path_{i}'] = res
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{n_paths} paths...")

    # Save to Parquet
    print(f"Saving to {output_path}...")
    final_df = pd.DataFrame(all_paths)
    final_df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    input_csv = "data/DAT_ASCII_EURGBP_M1_2025.csv"
    output_pq = "BB backtest/montecarlo/bootstrap_paths.parquet"
    generate_weekly_bootstrap(input_csv, output_pq)
