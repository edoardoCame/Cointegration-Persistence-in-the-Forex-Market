import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import cupy as cp
import cudf
import plotly.express as px
from system.data_feed import load_data, get_all_symbols
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../deliverables/03_extreme_deciles")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wide Search Grid (up to 1 month lookback, 2 days hold)
LOOKBACK_OPTS = [60, 240, 480, 1440, 2880, 10080, 20160, 43200]
HORIZON_OPTS = [15, 60, 240, 480, 1440, 2880]

def find_extreme_spread(df, symbol):
    """
    Finds the (Lookback, Horizon) combination that maximizes the 
    absolute spread between Top and Bottom deciles using GPU.
    """
    best_score = -1
    best_config = None
    best_means = None
    
    # Pre-calc Log Returns on GPU (as Series)
    log_ret_series = np.log(df['Close'] / df['Close'].shift(1))
    
    # We need to handle rolling windows. cudf rolling is fast.
    # We will loop but keep data on GPU.
    
    for lb in LOOKBACK_OPTS:
        lb_periods = int(lb / 15)
        # Momentum Signal (cudf Series)
        mom_series = log_ret_series.rolling(lb_periods).sum()
        
        for h in HORIZON_OPTS:
            h_periods = int(h / 15)
            # Future Return (cudf Series)
            fwd_series = log_ret_series.rolling(h_periods).sum().shift(-h_periods)
            
            # Drop NaNs - align
            # We can use .dropna() on a DataFrame of the two
            # Constructing a DF is cheap
            temp_df = cudf.DataFrame({'m': mom_series, 'f': fwd_series}).dropna()
            
            if len(temp_df) < 500: continue
            
            # Extract to CuPy for sorting
            m_vals = temp_df['m'].values
            f_vals = temp_df['f'].values
            
            # Sort indices by Momentum
            sorted_idx = cp.argsort(m_vals)
            
            # Reorder Future Returns based on Momentum Rank
            f_sorted = f_vals[sorted_idx]
            
            # Split into 10 Deciles
            n = len(f_sorted)
            chunk_size = n // 10
            
            # We only strictly need Top and Bottom for the score, 
            # but we need all means for the plot.
            # Efficiently reshape? If n is not divisible by 10, we drop remainder or handle it.
            # Truncating slightly to be divisible is faster and statistically fine for large N.
            n_trim = chunk_size * 10
            f_trimmed = f_sorted[:n_trim]
            
            # Reshape to (10, chunk_size) and take mean along axis 1
            decile_matrix = f_trimmed.reshape(10, chunk_size)
            decile_means = decile_matrix.mean(axis=1)
            
            # Score
            bottom = decile_means[0]
            top = decile_means[-1]
            spread = abs(top - bottom)
            
            if spread > best_score:
                best_score = spread
                best_config = (lb, h)
                # Store means on CPU for plotting later
                best_means = cp.asnumpy(decile_means)
                
    return best_config, best_score, best_means

def main():
    symbols = get_all_symbols()
    print(f"Searching for Extreme Decile Spreads on {len(symbols)} pairs...")
    
    results = []
    
    for sym in symbols:
        df = load_data(sym, '15min')
        if df is None: continue
        
        print(f"Scanning {sym}...", end='\r')
        config, score, data = find_extreme_spread(df, sym)
        
        if config:
            lb, h = config
            
            # Normalize to bps and create DataFrame
            data_bps = pd.DataFrame({
                'Decile': range(10),
                'Return_bps': data * 10000
            })
            
            title = (f"<b>{sym} MOST EXTREME: L:{lb}m -> H:{h}m</b><br>"
                     f"Max Spread: {score*10000:.1f} bps | Top vs Bottom")
            
            fig = px.bar(
                data_bps,
                x='Decile', y='Return_bps',
                title=title,
                labels={'Decile': 'Momentum Decile (0=Low, 9=High)', 'Return_bps': 'Future Return (bps)'},
                color='Return_bps',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            fig.update_layout(template='plotly_dark', showlegend=False)
            fig.add_hline(y=0, line_color="white", opacity=0.5)
            
            out_file = os.path.join(OUTPUT_DIR, f"extreme_{sym}.png")
            fig.write_image(out_file)
            
            results.append({'Symbol': sym, 'Lookback': lb, 'Horizon': h, 'Spread_bps': score*10000})
            
    print("\nGeneration Complete.")
    
    # Save Summary CSV
    res_df = pd.DataFrame(results).sort_values('Spread_bps', ascending=False)
    print("\nTop 5 Most Extreme Assets:")
    print(res_df.head(5))
    res_df.to_csv(os.path.join(OUTPUT_DIR, "extreme_summary.csv"), index=False)

if __name__ == "__main__":
    main()
