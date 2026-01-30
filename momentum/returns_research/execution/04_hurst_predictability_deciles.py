import sys
import os
import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.getcwd())

from returns_research.system import data_feed

def calculate_rolling_hurst_gpu(series, window, min_scales=4):
    """
    Calculates Rolling Hurst Exponent using Aggregated Variance Method on GPU. 
    
    Args:
        series: cudf.Series of prices
        window: Rolling window size in minutes
    """
    # Scales (lags) for variance calculation: 1, 2, 4, 8, ...
    # We need lags << window. For 1440 (1 day), 64 is safe (approx 22 points per window per lag? No.)
    # Window 1440. Lag 32. 1440/32 = 45 samples. Good.
    scales = [1, 2, 4, 8, 16, 32]
    if window >= 7200: # 5 days
        scales.append(64)
        
    log_scales = cp.log(cp.array(scales))
    X_mean = cp.mean(log_scales)
    SS_xx = cp.sum((log_scales - X_mean)**2)
    
    slope_numerator = cudf.Series(cp.zeros(len(series)), index=series.index)
    
    for m in scales:
        # Rolling Variance of (X_t - X_{t-m})
        # Note: We use diff(m) to get returns over m minutes
        diff_m = series.diff(m)
        
        # Rolling Variance over the window
        roll_var_m = diff_m.rolling(window).var()
        
        # Log of variance
        log_var_m = cp.log(roll_var_m.fillna(np.nan).values)
        
        # Regression term
        x_val = cp.log(m) - X_mean
        slope_numerator += (log_var_m * x_val)
        
    slope = slope_numerator / SS_xx
    hurst = slope / 2.0
    return hurst

def main():
    print("Loading EURCHF data...")
    df = data_feed.load_data('EURCHF', timeframe='1min')
    if df is None: return

    # --- Configuration ---
    WINDOW_PAST = 5 * 24 * 60  # 5 Days (7200 mins)
    WINDOW_FUTURE = 1 * 24 * 60 # 1 Day (1440 mins)
    
    print(f"Calculating Rolling Hurst (Past 5 Days)... Window={WINDOW_PAST}")
    hurst_past = calculate_rolling_hurst_gpu(df['Close'], window=WINDOW_PAST)
    
    print(f"Calculating Rolling Hurst (Future 1 Day)... Window={WINDOW_FUTURE}")
    # Calculate rolling 1-day Hurst normally (backward looking)
    hurst_1d_raw = calculate_rolling_hurst_gpu(df['Close'], window=WINDOW_FUTURE)
    
    # SHIFT it back by 1 day to make it the target for 't'
    # At time t, 'hurst_future' will contain the Hurst of [t, t+1day]
    # To get Hurst of [t, t+1d], we take the value at t+1d (which looks back to t) and put it at t.
    hurst_future = hurst_1d_raw.shift(-WINDOW_FUTURE)
    
    # Combine into a DataFrame for analysis
    # Drop NaNs
    data_gpu = cudf.DataFrame({
        'H_past': hurst_past,
        'H_future': hurst_future
    }).dropna()
    
    print(f"Valid Samples: {len(data_gpu)}")
    
    # --- Decile Analysis ---
    # Move to Pandas for qcut and plotting (easier API)
    data = data_gpu.to_pandas()
    
    # We bin 'H_past' into deciles and get the bins
    import pandas as pd
    data['decile_past'], bins = pd.qcut(data['H_past'], q=10, retbins=True, labels=False)
    
    # We want to check if H_future falls into the SAME bin as H_past.
    # We use pd.cut with the SAME bins derived from H_past to categorize H_future.
    # Note: bins[0] might need -inf and bins[-1] +inf to catch edge cases, 
    # but since H is bounded 0-1 usually, using the same bins is a fair test of "same numeric regime".
    # H_future might be outside the min/max of H_past, so we handle that.
    
    data['decile_future'] = pd.cut(data['H_future'], bins=bins, labels=False, include_lowest=True)
    
    # Define "Success" as remaining in the same decile
    # We need to handle NaNs in decile_future (if H_future is outside H_past historic range)
    # If it's outside range, it's definitely NOT in the same decile (unless edge deciles cover it? cut handles this)
    # decile_future will be NaN if outside bins.
    
    data['stayed_in_decile'] = (data['decile_past'] == data['decile_future']).astype(int)
    
    # Group by Decile Past
    results = data.groupby('decile_past')['stayed_in_decile'].mean()
    
    # Get mean H_past for each decile for labelling
    avg_h_past = data.groupby('decile_past')['H_past'].mean()
    
    # --- Plotting ---
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))
    
    # Create colors: Green to Red
    colors = plt.cm.RdYlGn(np.linspace(1, 0, 10)) 
    
    bars = plt.bar(results.index, results.values * 100, color=colors, alpha=0.9)
    
    plt.title('Regime Stasis: "If H is in Decile X today, will it be in Decile X tomorrow?"', fontsize=16)
    plt.xlabel('Decile of Past 5-Day Hurst Exponent\n(0 = Strongest MR, 9 = Strongest Trend)', fontsize=12)
    plt.ylabel('Probability of Remaining in Same Decile (%)', fontsize=12)
    
    # X-Axis Labels with Range
    labels = []
    for i in range(10):
        # bins[i] to bins[i+1]
        labels.append(f"D{i}\n{bins[i]:.2f}-{bins[i+1]:.2f}")
        
    plt.xticks(range(10), labels, rotation=0, fontsize=9)
    
    # Random Chance line? 1/10 = 10%
    plt.axhline(10, color='white', linestyle='--', alpha=0.5, label='Random Chance (10%)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    
    out_dir = "returns_research/deliverables/04_hurst_predictability"
    plt.savefig(f"{out_dir}/decile_predictability.png", dpi=300, bbox_inches='tight')
    print("Saved decile_predictability.png")
    plt.close()
    
    # Scatter plot for detail?
    # Let's add a correlation plot
    plt.figure(figsize=(10, 8))
    # Sample for scatter
    if len(data) > 20000:
        sample = data.sample(20000)
    else:
        sample = data
        
    sns.regplot(x=sample['H_past'], y=sample['H_future'], 
                scatter_kws={'alpha':0.05, 's':2, 'color':'cyan'}, 
                line_kws={'color':'red'})
    
    plt.title(f'Correlation: Past 5-Day vs Future 1-Day Hurst\nCorr: {sample["H_past"].corr(sample["H_future"]):.3f}', fontsize=14)
    plt.xlabel('Past 5-Day Hurst')
    plt.ylabel('Future 1-Day Hurst')
    plt.grid(True, alpha=0.2)
    
    plt.savefig(f"{out_dir}/hurst_correlation.png", dpi=300, bbox_inches='tight')
    print("Saved hurst_correlation.png")
    plt.close()

if __name__ == "__main__":
    main()
