import cudf
import numpy as np
from numba import cuda
import math

# 1. CUDA Kernel for Grid Search
@cuda.jit
def bb_grid_search_kernel(close, means, stds, std_mults, commission, results):
    lb_idx = cuda.blockIdx.x
    sm_idx = cuda.threadIdx.x
    
    if lb_idx < means.shape[0] and sm_idx < std_mults.shape[0]:
        cur_means = means[lb_idx]
        cur_stds = stds[lb_idx]
        std_mult = std_mults[sm_idx]
        
        current_pos = 0 
        total_ret = 0.0
        n_trades = 0
        n = len(close)
        
        for i in range(1, n):
            price = close[i]
            prev_price = close[i-1]
            m = cur_means[i]
            s = cur_stds[i]
            
            if s <= 0 or math.isnan(m) or math.isnan(s): continue
                
            if current_pos == 1: total_ret += (price - prev_price)
            elif current_pos == -1: total_ret += (prev_price - price)
            
            upper = m + s * std_mult
            lower = m - s * std_mult
            
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
                    
        results[lb_idx, sm_idx, 0] = total_ret
        results[lb_idx, sm_idx, 1] = float(n_trades)

# 2. CUDA Kernel for Final Path
@cuda.jit
def bb_final_path_kernel(close, m, s, std_mult, commission, net_returns):
    if cuda.grid(1) == 0:
        current_pos = 0
        for i in range(1, len(close)):
            price = close[i]
            prev_price = close[i-1]
            ret = 0.0
            if current_pos == 1: ret = (price - prev_price)
            elif current_pos == -1: ret = (prev_price - price)
            
            upper = m[i] + s[i] * std_mult
            lower = m[i] - s[i] * std_mult
            
            comm = 0.0
            if current_pos == 0:
                if price < lower:
                    current_pos = 1
                    comm = commission
                elif price > upper:
                    current_pos = -1
                    comm = commission
            elif current_pos == 1:
                if price >= m[i]:
                    current_pos = 0
                    comm = commission
            elif current_pos == -1:
                if price <= m[i]:
                    current_pos = 0
                    comm = commission
            net_returns[i] = ret - comm
