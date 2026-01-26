import cudf
import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
import os
import math
import time
import pandas as pd
from tqdm import tqdm
import sys

# Import shared kernels
sys.path.append(os.path.join(os.getcwd(), "BB backtest/scripts"))
from bb_gpu_lib import bb_batch_grid_search_kernel, bb_mc_oos_kernel

def run_montecarlo_sim(pq_path, batch_size=1000):
    """Simulazione batch ottimizzata"""
    if not os.path.exists(pq_path):
        print(f"Error: {pq_path} not found. Run bootstrap_gen.py first.")
        return

    import pyarrow.parquet as pq
    table_meta = pq.read_metadata(pq_path)
    path_cols = [c for c in table_meta.schema.names if c.startswith('path_')]
    n_total_paths = len(path_cols)
    
    print(f"Starting Monte Carlo Simulation on {n_total_paths} paths...")
    
    lookbacks = np.arange(1000, 20001, 2000).astype(np.float64)
    std_mults = np.arange(1.5, 5.1, 0.25).astype(np.float64)
    commission = 0.5 * 0.0001
    
    all_oos_pips = []
    start_total = time.time()
    
    pbar = tqdm(range(0, n_total_paths, batch_size), desc="MC Simulation")
    for i in pbar:
        batch_cols = path_cols[i : i + batch_size]
        actual_batch_size = len(batch_cols)
        pbar.set_postfix({"batch": i//batch_size + 1, "paths": i + actual_batch_size})
        
        df_batch = cudf.read_parquet(pq_path, columns=batch_cols)
        split_idx = int(len(df_batch) * 0.7)
        all_paths_gpu = cp.stack([df_batch[c].values for c in batch_cols])
        
        # Grid Search
        d_win_indices = cuda.to_device(np.array([[0, split_idx]] * actual_batch_size))
        d_results = cuda.device_array((actual_batch_size, len(lookbacks), len(std_mults), 2), dtype=np.float64)
        bb_batch_grid_search_kernel[(actual_batch_size, len(lookbacks)), len(std_mults)](
            all_paths_gpu, d_win_indices, cuda.to_device(lookbacks), cuda.to_device(std_mults), commission, d_results
        )
        cuda.synchronize()
        
        # Best Params
        batch_results = d_results.copy_to_host()
        best_lbs, best_sms = [], []
        for p in range(actual_batch_size):
            res = batch_results[p]
            masked_ret = np.where(res[:,:,1] >= 20, res[:,:,0], -1e9)
            idx = np.unravel_index(np.argmax(masked_ret), masked_ret.shape)
            best_lbs.append(lookbacks[idx[0]]); best_sms.append(std_mults[idx[1]])
            
        # OOS
        d_oos_gains = cuda.device_array(actual_batch_size, dtype=np.float64)
        bb_mc_oos_kernel[actual_batch_size, 1](
            all_paths_gpu, cuda.to_device(np.array(best_lbs)), cuda.to_device(np.array(best_sms)), commission, split_idx, d_oos_gains
        )
        cuda.synchronize()
        all_oos_pips.extend((d_oos_gains.copy_to_host() / 0.0001).tolist())
        
        del df_batch, all_paths_gpu, d_results, d_oos_gains
        cp._default_memory_pool.free_all_blocks()

    # Save results
    res_dir = "BB backtest/results/montecarlo"
    os.makedirs(res_dir, exist_ok=True)
    all_oos_pips = np.array(all_oos_pips)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_oos_pips, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.axvline(np.median(all_oos_pips), color='red', linestyle='--', label=f'Median: {np.median(all_oos_pips):.1f}')
    plt.title(f"MC OOS Distribution ({n_total_paths} paths)")
    plt.xlabel("Net Pips")
    plt.legend(); plt.savefig(f"{res_dir}/oos_distribution.png")
    
    with open(f"{res_dir}/mc_stats.txt", "w") as f:
        f.write(f"Monte Carlo Batched Results ({n_total_paths} paths)\n")
        f.write(f"Mean OOS: {all_oos_pips.mean():.2f}\nMedian OOS: {np.median(all_oos_pips):.2f}\n")
        f.write(f"Positive: {(all_oos_pips > 0).sum() / n_total_paths * 100:.1f}%\n")

    print(f"\nMonte Carlo su {n_total_paths} path completato in {time.time() - start_total:.2f}s!")
    print(f"Media Finale: {all_oos_pips.mean():.1f} pips")
    print(f"Mediana Finale: {np.median(all_oos_pips):.1f} pips")

if __name__ == "__main__":
    pq_path = "data/bootstrap_paths.parquet"
    run_montecarlo_sim(pq_path)