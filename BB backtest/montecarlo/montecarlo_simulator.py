import cudf
import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
import os
import math
import time
from tqdm import tqdm

# --- CUDA KERNELS ---

@cuda.jit
def bb_mc_batch_grid_search_kernel(all_paths, lookbacks, std_mults, commission, split_idx, results):
    path_idx = cuda.blockIdx.x
    lb_idx = cuda.blockIdx.y
    sm_idx = cuda.threadIdx.x
    
    if path_idx >= all_paths.shape[0] or lb_idx >= lookbacks.shape[0] or sm_idx >= std_mults.shape[0]:
        return
        
    lb = int(lookbacks[lb_idx])
    sm = std_mults[sm_idx]
    
    current_pos = 0 
    total_ret = 0.0
    n_trades = 0
    
    s1 = 0.0
    s2 = 0.0
    for i in range(lb):
        val = all_paths[path_idx, i]
        s1 += val
        s2 += val * val
        
    for i in range(1, split_idx):
        price = all_paths[path_idx, i]
        prev_price = all_paths[path_idx, i-1]
        
        if current_pos == 1: total_ret += (price - prev_price)
        elif current_pos == -1: total_ret += (prev_price - price)
        
        if i >= lb - 1:
            if i > lb - 1:
                old_val = all_paths[path_idx, i - lb]
                new_val = price
                s1 += (new_val - old_val)
                s2 += (new_val * new_val - old_val * old_val)
            
            m = s1 / lb
            v = (s2 / lb) - (m * m)
            s = math.sqrt(v) if v > 0 else 0.0
            
            if s <= 0: continue
            
            upper = m + s * sm
            lower = m - s * sm
            
            if current_pos == 0:
                if price < lower:
                    current_pos = 1
                    total_ret -= commission
                    n_trades += 1
                elif price > upper:
                    current_pos = -1
                    total_ret -= commission
                    n_trades += 1
            elif current_pos == 1:
                if price >= m:
                    current_pos = 0
                    total_ret -= commission
            elif current_pos == -1:
                if price <= m:
                    current_pos = 0
                    total_ret -= commission
                    
    results[path_idx, lb_idx, sm_idx, 0] = total_ret
    results[path_idx, lb_idx, sm_idx, 1] = float(n_trades)

@cuda.jit
def bb_mc_batch_oos_kernel(all_paths, best_lbs, best_sms, commission, split_idx, oos_gains):
    path_idx = cuda.grid(1)
    if path_idx >= all_paths.shape[0]: return
    
    lb = int(best_lbs[path_idx])
    sm = best_sms[path_idx]
    n_samples = all_paths.shape[1]
    
    current_pos = 0
    total_oos_ret = 0.0
    
    s1 = 0.0
    s2 = 0.0
    for i in range(lb):
        val = all_paths[path_idx, i]
        s1 += val
        s2 += val * val
        
    for i in range(1, n_samples):
        price = all_paths[path_idx, i]
        prev_price = all_paths[path_idx, i-1]
        
        ret = 0.0
        if current_pos == 1: ret = (price - prev_price)
        elif current_pos == -1: ret = (prev_price - price)
        
        comm = 0.0
        if i >= lb - 1:
            if i > lb - 1:
                old_val = all_paths[path_idx, i - lb]
                new_val = price
                s1 += (new_val - old_val)
                s2 += (new_val * new_val - old_val * old_val)
            
            m = s1 / lb
            v = (s2 / lb) - (m * m)
            s = math.sqrt(v) if v > 0 else 0.0
            
            upper = m + s * sm
            lower = m - s * sm
            
            if current_pos == 0:
                if price < lower:
                    current_pos = 1
                    comm = commission
                elif price > upper:
                    current_pos = -1
                    comm = commission
            elif current_pos == 1:
                if price >= m:
                    current_pos = 0
                    comm = commission
            elif current_pos == -1:
                if price <= m:
                    current_pos = 0
                    comm = commission
        
        if i > split_idx:
            total_oos_ret += (ret - comm)
            
    oos_gains[path_idx] = total_oos_ret

# --- MAIN RUNNER ---

def run_montecarlo_optimized():
    pq_path = "BB backtest/montecarlo/bootstrap_paths.parquet"
    if not os.path.exists(pq_path):
        print(f"Error: {pq_path} not trovato.")
        return

    # Leggiamo i nomi delle colonne per fare batching
    import pyarrow.parquet as pq
    table_meta = pq.read_metadata(pq_path)
    all_cols = table_meta.schema.names
    path_cols = [c for c in all_cols if c.startswith('path_')]
    n_total_paths = len(path_cols)
    
    print(f"Trovati {n_total_paths} path. Avvio elaborazione in batch...")

    lookbacks = np.arange(1000, 20001, 2000).astype(np.float64)
    std_mults = np.arange(1.5, 5.1, 0.25).astype(np.float64)
    commission = 0.5 * 0.0001
    
    batch_size = 1000
    all_oos_pips = []
    
    start_total = time.time()
    
    pbar = tqdm(range(0, n_total_paths, batch_size), desc="Processing Batches")
    for i in pbar:
        batch_cols = path_cols[i : i + batch_size]
        actual_batch_size = len(batch_cols)
        pbar.set_postfix({"batch": i//batch_size + 1, "paths": i + actual_batch_size})
        
        # Carica solo le colonne del batch
        df_batch = cudf.read_parquet(pq_path, columns=batch_cols)
        n_samples = len(df_batch)
        split_idx = int(n_samples * 0.7)
        
        # Trasferimento a CuPy matrix
        all_paths_gpu = cp.stack([df_batch[c].values for c in batch_cols])
        
        d_lookbacks = cuda.to_device(lookbacks)
        d_std_mults = cuda.to_device(std_mults)
        d_results = cuda.device_array((actual_batch_size, len(lookbacks), len(std_mults), 2), dtype=np.float64)
        
        # Grid Search
        grid = (actual_batch_size, len(lookbacks))
        block = (len(std_mults))
        bb_mc_batch_grid_search_kernel[grid, block](all_paths_gpu, d_lookbacks, d_std_mults, commission, split_idx, d_results)
        cuda.synchronize()
        
        # Ottimizzazione Parametri (su Host)
        batch_results = d_results.copy_to_host()
        best_lbs = np.zeros(actual_batch_size)
        best_sms = np.zeros(actual_batch_size)
        
        for p in range(actual_batch_size):
            res = batch_results[p]
            masked_ret = np.where(res[:,:,1] >= 20, res[:,:,0], -1e9)
            best_idx = np.unravel_index(np.argmax(masked_ret), masked_ret.shape)
            best_lbs[p] = lookbacks[best_idx[0]]
            best_sms[p] = std_mults[best_idx[1]]
            
        # OOS Validation
        d_best_lbs = cuda.to_device(best_lbs)
        d_best_sms = cuda.to_device(best_sms)
        d_oos_gains = cuda.device_array(actual_batch_size, dtype=np.float64)
        
        bb_mc_batch_oos_kernel[actual_batch_size, 1](all_paths_gpu, d_best_lbs, d_best_sms, commission, split_idx, d_oos_gains)
        cuda.synchronize()
        
        batch_oos_pips = d_oos_gains.copy_to_host() / 0.0001
        all_oos_pips.extend(batch_oos_pips.tolist())
        
        # Pulizia esplicita per il prossimo batch
        del df_batch, all_paths_gpu, d_results, d_oos_gains
        cp._default_memory_pool.free_all_blocks()

    all_oos_pips = np.array(all_oos_pips)
    
    # --- RISULTATI FINALI ---
    results_dir = "BB backtest/montecarlo/results"
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.hist(all_oos_pips, bins=50, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(np.median(all_oos_pips), color='red', linestyle='dashed', label=f'Median: {np.median(all_oos_pips):.1f}')
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Monte Carlo ({n_total_paths} paths) - Batched GPU Execution")
    plt.xlabel("Net Pips")
    plt.legend()
    plt.savefig(f"{results_dir}/oos_distribution.png")
    
    with open(f"{results_dir}/mc_stats.txt", "w") as f:
        f.write(f"Monte Carlo Batched Results ({n_total_paths} paths)\n")
        f.write(f"Tempo totale: {time.time() - start_total:.2f}s\n")
        f.write(f"Mean OOS Gain: {all_oos_pips.mean():.2f} pips\n")
        f.write(f"Median OOS Gain: {np.median(all_oos_pips):.2f} pips\n")
        f.write(f"Positive Outcomes: {((all_oos_pips > 0).sum() / n_total_paths * 100):.1f}%")
        f.write(f"Worst Case: {all_oos_pips.min():.2f} pips\n")
        f.write(f"Best Case: {all_oos_pips.max():.2f} pips\n")

    print(f"\nMonte Carlo su {n_total_paths} path completato in {time.time() - start_total:.2f}s!")
    print(f"Media Finale: {all_oos_pips.mean():.1f} pips")
    print(f"Mediana Finale: {np.median(all_oos_pips):.1f} pips")

if __name__ == "__main__":
    run_montecarlo_optimized()