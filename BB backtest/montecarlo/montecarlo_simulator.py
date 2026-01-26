import cudf
import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm

# --- CUDA KERNELS ---

@cuda.jit
def bb_mc_batch_grid_search_kernel(all_paths, lookbacks, std_mults, commission, split_idx, results):
    """
    Grid Search parallelo su TUTTI i path contemporaneamente.
    All_paths: (n_paths, n_samples)
    Results: (n_paths, n_lookbacks, n_std_mults, 2) -> [return, trades]
    """
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
    
    # Inizializzazione sliding window (somme per i primi lb elementi)
    s1 = 0.0
    s2 = 0.0
    for i in range(lb):
        val = all_paths[path_idx, i]
        s1 += val
        s2 += val * val
        
    # Ciclo principale In-Sample
    for i in range(1, split_idx):
        price = all_paths[path_idx, i]
        prev_price = all_paths[path_idx, i-1]
        
        # 1. Aggiorna equity dalla posizione precedente
        if current_pos == 1: total_ret += (price - prev_price)
        elif current_pos == -1: total_ret += (prev_price - price)
        
        # 2. Aggiorna finestra e calcola segnali
        if i >= lb - 1:
            if i > lb - 1:
                # Sliding window update: aggiungi nuovo, togli vecchio
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
    """Esegue il backtest OOS finale usando i parametri migliori trovati per ogni path"""
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
        
        # Accumula solo se siamo in OOS
        if i > split_idx:
            total_oos_ret += (ret - comm)
            
    oos_gains[path_idx] = total_oos_ret

# --- MAIN RUNNER ---

def run_montecarlo_optimized():
    pq_path = "BB backtest/montecarlo/bootstrap_paths.parquet"
    if not os.path.exists(pq_path):
        print("Error: bootstrap_paths.parquet non trovato.")
        return

    print("Caricamento path in GPU...")
    df_paths = cudf.read_parquet(pq_path)
    path_cols = [c for c in df_paths.columns if c.startswith('path_')]
    
    # Prepariamo i dati come singola matrice 2D (n_paths, n_samples)
    all_paths_gpu = cp.stack([df_paths[c].values for c in path_cols])
    n_paths = all_paths_gpu.shape[0]
    n_samples = all_paths_gpu.shape[1]
    
    lookbacks = np.arange(1000, 20001, 2000).astype(np.float64)
    std_mults = np.arange(1.5, 5.1, 0.25).astype(np.float64)
    commission = 0.5 * 0.0001
    split_idx = int(n_samples * 0.7)
    
    d_lookbacks = cuda.to_device(lookbacks)
    d_std_mults = cuda.to_device(std_mults)
    d_results = cuda.device_array((n_paths, len(lookbacks), len(std_mults), 2), dtype=np.float64)
    
    print(f"Lancio Batch Grid Search su {n_paths} path...")
    # Grid: (n_paths, n_lookbacks), Threads: n_std_mults
    grid = (n_paths, len(lookbacks))
    block = (len(std_mults))
    
    import time
    start = time.time()
    bb_mc_batch_grid_search_kernel[grid, block](all_paths_gpu, d_lookbacks, d_std_mults, commission, split_idx, d_results)
    cuda.synchronize()
    grid_time = time.time() - start
    print(f"Grid Search completata in {grid_time:.2f}s")
    
    # Trova i parametri migliori per ogni path (su Host per semplicità)
    results = d_results.copy_to_host()
    best_lbs = np.zeros(n_paths)
    best_sms = np.zeros(n_paths)
    
    for p in range(n_paths):
        path_res = results[p]
        # Filtro minimo 20 trade
        masked_ret = np.where(path_res[:,:,1] >= 20, path_res[:,:,0], -1e9)
        best_idx = np.unravel_index(np.argmax(masked_ret), masked_ret.shape)
        best_lbs[p] = lookbacks[best_idx[0]]
        best_sms[p] = std_mults[best_idx[1]]
        
    # Lancio Validation OOS finale
    d_best_lbs = cuda.to_device(best_lbs)
    d_best_sms = cuda.to_device(best_sms)
    d_oos_gains = cuda.device_array(n_paths, dtype=np.float64)
    
    print("Lancio OOS Validation parallela...")
    bb_mc_batch_oos_kernel[n_paths, 1](all_paths_gpu, d_best_lbs, d_best_sms, commission, split_idx, d_oos_gains)
    cuda.synchronize()
    
    oos_pips = d_oos_gains.copy_to_host() / 0.0001
    
    # --- RISULTATI ---
    results_dir = "BB backtest/montecarlo/results"
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.hist(oos_pips, bins=25, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(oos_pips.mean(), color='red', linestyle='dashed', label=f'Mean: {oos_pips.mean():.1f}')
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Monte Carlo Ultra-Fast: OOS Results ({n_paths} paths)")
    plt.xlabel("Net Pips")
    plt.legend()
    plt.savefig(f"{results_dir}/oos_distribution.png")
    
    with open(f"{results_dir}/mc_stats.txt", "w") as f:
        f.write(f"Monte Carlo Batch Optimized Results ({n_paths} paths)\n")
        f.write(f"Tempo esecuzione Kernel: {grid_time:.2f}s\n")
        f.write(f"Mean OOS Gain: {oos_pips.mean():.2f} pips\n")
        f.write(f"Positive Outcomes: {((oos_pips > 0).sum() / n_paths * 100):.1f}%")
        f.write(f"Worst Case: {oos_pips.min():.2f} pips\n")
        f.write(f"Best Case: {oos_pips.max():.2f} pips\n")

    print(f"\nMonte Carlo completato in {time.time() - start:.2f}s!")
    print(f"Media: {oos_pips.mean():.1f} pips")

if __name__ == "__main__":
    run_montecarlo_optimized()
