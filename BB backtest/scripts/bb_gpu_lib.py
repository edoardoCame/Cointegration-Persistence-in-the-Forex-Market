import numpy as np
from numba import cuda
import math

# --- KERNEL 1: BATCH GRID SEARCH (Per MC e WFA) ---
@cuda.jit
def bb_batch_grid_search_kernel(prices, window_indices, lookbacks, std_mults, commission, results):
    """
    Grid Search ottimizzata con Sliding Window.
    window_indices: (n_windows, 2) [start, end]
    results: (n_windows, n_lookbacks, n_std_mults, 2)
    """
    w_idx = cuda.blockIdx.x
    lb_idx = cuda.blockIdx.y
    sm_idx = cuda.threadIdx.x
    
    if w_idx >= window_indices.shape[0] or lb_idx >= lookbacks.shape[0] or sm_idx >= std_mults.shape[0]:
        return
        
    start_idx = window_indices[w_idx, 0]
    end_idx = window_indices[w_idx, 1]
    lb = int(lookbacks[lb_idx])
    sm = std_mults[sm_idx]
    
    current_pos = 0 
    total_ret = 0.0
    n_trades = 0
    
    s1 = 0.0
    s2 = 0.0
    # Warmup
    for i in range(start_idx, start_idx + lb):
        val = prices[i]
        s1 += val
        s2 += val * val
        
    for i in range(start_idx + 1, end_idx):
        price = prices[i]
        prev_price = prices[i-1]
        
        if current_pos == 1: total_ret += (price - prev_price)
        elif current_pos == -1: total_ret += (prev_price - price)
        
        if i >= start_idx + lb - 1:
            if i > start_idx + lb - 1:
                old_val = prices[i - lb]
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
                    
    results[w_idx, lb_idx, sm_idx, 0] = total_ret
    results[w_idx, lb_idx, sm_idx, 1] = float(n_trades)

# --- KERNEL 2: BATCH OOS VALIDATION (Per MC) ---
@cuda.jit
def bb_mc_oos_kernel(all_paths, best_lbs, best_sms, commission, split_idx, oos_gains):
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
        ret = (price - prev_price) if current_pos == 1 else (prev_price - price) if current_pos == -1 else 0.0
        
        old_val = all_paths[path_idx, i - lb]
        new_val = price
        s1 += (new_val - old_val)
        s2 += (new_val * new_val - old_val * old_val)
        
        m = s1 / lb
        v = (s2 / lb) - (m * m)
        s = math.sqrt(v) if v > 0 else 0.0
        
        upper = m + s * sm
        lower = m - s * sm
        comm = 0.0
        if current_pos == 0:
            if price < lower: current_pos = 1; comm = commission
            elif price > upper: current_pos = -1; comm = commission
        elif current_pos == 1:
            if price >= m: current_pos = 0; comm = commission
        elif current_pos == -1:
            if price <= m: current_pos = 0; comm = commission
        
        if i > split_idx:
            total_oos_ret += (ret - comm)
            
    oos_gains[path_idx] = total_oos_ret

# --- KERNEL 3: CONTINUOUS OOS (Per WFA) ---
@cuda.jit
def bb_wfa_oos_kernel(prices, week_params, week_indices, commission, net_returns):
    if cuda.grid(1) == 0:
        current_pos = 0
        for w in range(len(week_indices)):
            lb = int(week_params[w, 0])
            sm = week_params[w, 1]
            start_idx = int(week_indices[w, 0])
            end_idx = int(week_indices[w, 1])
            
            s1, s2 = 0.0, 0.0
            for i in range(start_idx - lb, start_idx):
                v = prices[i]; s1 += v; s2 += v*v
                
            for i in range(start_idx, end_idx):
                price, prev_price = prices[i], prices[i-1]
                ret = (price - prev_price) if current_pos == 1 else (prev_price - price) if current_pos == -1 else 0.0
                
                old_val, new_val = prices[i - lb], price
                s1 += (new_val - old_val); s2 += (new_val*new_val - old_val*old_val)
                m = s1 / lb; v_sq = (s2 / lb) - (m * m)
                s = math.sqrt(v_sq) if v_sq > 0 else 0.0
                
                upper, lower, comm = m + s*sm, m - s*sm, 0.0
                if current_pos == 0:
                    if price < lower: current_pos = 1; comm = commission
                    elif price > upper: current_pos = -1; comm = commission
                elif current_pos == 1:
                    if price >= m: current_pos = 0; comm = commission
                elif current_pos == -1:
                    if price <= m: current_pos = 0; comm = commission
                
                net_returns[i] = ret - comm