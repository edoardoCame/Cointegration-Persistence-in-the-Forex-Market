import sys
import os
import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to sys.path
sys.path.append(os.getcwd())

from returns_research.system import data_feed

def calculate_rolling_theta(series, window=4320):
    """
    Calculates rolling Theta from OU process using GPU.
    Theta = -ln(beta) where x(t+1) = alpha + beta*x(t) + e
    """
    y = series
    x = series.shift(1)
    
    # Rolling Sums (GPU)
    sum_x = x.rolling(window).sum()
    sum_y = y.rolling(window).sum()
    sum_xx = (x**2).rolling(window).sum()
    sum_xy = (x*y).rolling(window).sum()
    
    N = window
    numerator = N * sum_xy - sum_x * sum_y
    denominator = N * sum_xx - sum_x**2
    
    beta = numerator / denominator
    
    # Theta calculation
    # We only care about valid mean reverting theta.
    # If beta <= 0 or beta >= 1, theta is undefined or 0 for our purpose.
    # We handle this by replacing invalid betas with NaN, then filling.
    
    # Convert to cupy to use log
    beta_vals = beta.fillna(np.nan).values
    
    # Theta = -ln(beta)
    # This will be NaN for beta <= 0
    # This will be negative (or 0) for beta >= 1. 
    # Realistically, for MR, 0 < beta < 1, so theta > 0.
    
    theta = -cp.log(beta_vals)
    
    return cudf.Series(theta, index=series.index)

def main():
    print("Loading EURCHF data...")
    df = data_feed.load_data('EURCHF', timeframe='1min')
    if df is None: return

    # --- Configuration ---
    WINDOW_PAST = 4 * 24 * 60   # 4 Days (5760 mins)
    WINDOW_FUTURE = 1 * 24 * 60 # 1 Day (1440 mins) - Prediction Horizon
    
    print(f"Calculating Rolling Theta (Past 4 Days)... Window={WINDOW_PAST}")
    theta_past = calculate_rolling_theta(df['Close'], window=WINDOW_PAST)
    
    print(f"Calculating Rolling Theta (Future 1 Day)... Window={WINDOW_FUTURE}")
    theta_future_raw = calculate_rolling_theta(df['Close'], window=WINDOW_FUTURE)
    
    # Shift future back to align with 't'
    theta_future = theta_future_raw.shift(-WINDOW_FUTURE)
    
    # Combine
    data_gpu = cudf.DataFrame({
        'theta_past': theta_past,
        'theta_future': theta_future
    })
    
    # Filter for valid regimes
    # We are interested in the predictability of Theta when it exists.
    # If Theta is NaN (Random Walk/Trend), we can either drop or treat as "Zero Reversion".
    # Let's drop NaN to focus on "When there is MR, how stable is it?"
    data_gpu = data_gpu.dropna()
    
    # Also filter out negative theta (momentum) if any leaked through math
    data_gpu = data_gpu[data_gpu['theta_past'] > 0]
    data_gpu = data_gpu[data_gpu['theta_future'] > 0]
    
    print(f"Valid MR Samples: {len(data_gpu)}")
    
    # Move to Pandas
    data = data_gpu.to_pandas()
    
    # --- Decile Analysis ---
    # Bin theta_past
    # Decile 9 = Highest Theta (Strongest Spring)
    # Decile 0 = Lowest Theta (Weakest Spring)
    data['decile_past'], bins = pd.qcut(data['theta_past'], q=10, retbins=True, labels=False)
    
    # Categorize theta_future using same bins
    data['decile_future'] = pd.cut(data['theta_future'], bins=bins, labels=False, include_lowest=True)
    
    # Success: Stay in same decile
    data['stayed_in_decile'] = (data['decile_past'] == data['decile_future']).astype(int)
    
    # Aggregation
    results = data.groupby('decile_past')['stayed_in_decile'].mean()
    avg_theta = data.groupby('decile_past')['theta_past'].mean()
    
    # --- Plotting ---
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))
    
    # Color map: Red (Low Theta/Weak) -> Green (High Theta/Strong)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, 10))
    
    bars = plt.bar(results.index, results.values * 100, color=colors, alpha=0.9)
    
    plt.title('Physics of Reversion: "If Past Theta (4d) was X, will Tomorrow be X?"', fontsize=16)
    plt.xlabel('Decile of Past 4-Day Theta\n(0 = Weakest Pull, 9 = Strongest Pull)', fontsize=12)
    plt.ylabel('Probability of Remaining in Same Intensity Regime (%)', fontsize=12)
    
    # X-Axis Labels
    labels = []
    for i in range(10):
        # Format scientific notation if needed, but usually theta is like 0.0001
        labels.append(f"D{i}\n{bins[i]*1000:.2f}-{bins[i+1]*1000:.2f}e-3")
        
    plt.xticks(range(10), labels, rotation=0, fontsize=9)
    
    # Random Chance (10%)
    plt.axhline(10, color='white', linestyle='--', alpha=0.5, label='Random Chance (10%)')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
                 
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    
    out_dir = "returns_research/deliverables/04_hurst_predictability"
    plt.savefig(f"{out_dir}/theta_decile_predictability.png", dpi=300, bbox_inches='tight')
    print("Saved theta_decile_predictability.png")
    plt.close()
    
    # Correlation Plot
    plt.figure(figsize=(10, 8))
    if len(data) > 20000:
        sample = data.sample(20000)
    else:
        sample = data
        
    sns.regplot(x=sample['theta_past'], y=sample['theta_future'],
                scatter_kws={'alpha':0.05, 's':2, 'color':'yellow'},
                line_kws={'color':'white'})
                
    plt.title(f'Correlation: Past Theta (4d) vs Future Theta (1d)\nCorr: {sample["theta_past"].corr(sample["theta_future"]):.3f}', fontsize=14)
    plt.xlabel('Past Theta (4d)')
    plt.ylabel('Future Theta (1d)')
    plt.grid(True, alpha=0.2)
    
    plt.savefig(f"{out_dir}/theta_correlation.png", dpi=300, bbox_inches='tight')
    print("Saved theta_correlation.png")
    plt.close()

if __name__ == "__main__":
    main()
