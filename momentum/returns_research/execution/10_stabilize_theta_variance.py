import sys
import os
import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Add project root to sys.path
sys.path.append(os.getcwd())

from returns_research.system import data_feed

def calculate_rolling_theta(series, window=4320):
    """Calculates rolling Theta from OU process using GPU."""
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
    beta_vals = beta.fillna(np.nan).values
    theta = -cp.log(beta_vals)
    return cudf.Series(theta, index=series.index)

def main():
    print("Loading EURCHF data...")
    df = data_feed.load_data('EURCHF', timeframe='1min')
    if df is None: return

    # --- Configuration ---
    WINDOW_PAST = 3 * 24 * 60
    WINDOW_FUTURE = 1 * 24 * 60
    
    print("Calculating Rolling Theta...")
    theta_past = calculate_rolling_theta(df['Close'], window=WINDOW_PAST)
    theta_future_raw = calculate_rolling_theta(df['Close'], window=WINDOW_FUTURE)
    theta_future = theta_future_raw.shift(-WINDOW_FUTURE)
    
    # Combine and Filter
    data_gpu = cudf.DataFrame({
        'x': theta_past,
        'y': theta_future
    }).dropna()
    
    # Filter positive theta
    data_gpu = data_gpu[data_gpu['x'] > 0]
    data_gpu = data_gpu[data_gpu['y'] > 0]
    
    # Sample for plotting and analysis (too many points kill matplotlib/scipy)
    # 20k points is enough for distribution analysis
    if len(data_gpu) > 20000:
        data = data_gpu.to_pandas().sample(20000, random_state=42)
    else:
        data = data_gpu.to_pandas()
        
    print(f"Analyzing {len(data)} samples...")
    
    x_raw = data['x'].values
    y_raw = data['y'].values
    
    # --- Transformations ---
    
    # 1. Log Transformation
    x_log = np.log(x_raw)
    y_log = np.log(y_raw)
    
    # 2. Sqrt Transformation
    x_sqrt = np.sqrt(x_raw)
    y_sqrt = np.sqrt(y_raw)
    
    # 3. Box-Cox
    # Box-Cox requires 1D array. We optimize lambda on X (predictor) or Y?
    # Usually we transform Y to stabilize variance, but here X and Y are the same variable at different times.
    # It makes sense to apply the SAME transform to both.
    # We'll find optimal lambda for the pooled data or just X. Let's use X.
    x_boxcox, lmbda = stats.boxcox(x_raw)
    y_boxcox = stats.boxcox(y_raw, lmbda=lmbda) # Use same lambda for consistency
    print(f"Optimal Box-Cox Lambda: {lmbda:.4f}")
    
    # --- WLS Analysis ---
    # We want to regress y_raw ~ x_raw but weight by inverse variance.
    # Estimate variance of residuals from OLS first.
    
    # OLS
    X_ols = sm.add_constant(x_raw)
    model_ols = sm.OLS(y_raw, X_ols).fit()
    residuals = model_ols.resid
    
    # Estimate variance structure: Regress squared residuals on X
    # resid^2 = a + b*X (or b*X^2)
    # Let's smooth the squared residuals to get a variance function
    # Bin X and calculate variance in bins
    bins = np.linspace(x_raw.min(), x_raw.max(), 20)
    bin_indices = np.digitize(x_raw, bins)
    
    # Calculate variance per bin
    bin_vars = {}
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 10:
            bin_vars[i] = np.var(residuals[mask])
        else:
            bin_vars[i] = 1.0 # Default fallback
            
    # Assign weights: w_i = 1 / var(bin(x_i))
    weights = np.array([1.0 / bin_vars.get(idx, 1.0) for idx in bin_indices])
    
    # WLS
    model_wls = sm.WLS(y_raw, X_ols, weights=weights).fit()
    print(f"OLS R-squared: {model_ols.rsquared:.4f}")
    print(f"WLS R-squared: {model_wls.rsquared:.4f}") # Weighted R2
    
    # --- Plotting ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    def plot_scatter(ax, x, y, title, color):
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.1, 's':2, 'color':color}, line_kws={'color':'white'})
        corr = np.corrcoef(x, y)[0,1]
        ax.set_title(f"{title}\nPearson Corr: {corr:.4f}", fontsize=12)
        ax.grid(True, alpha=0.2)
        
    # 1. Raw Data (OLS)
    plot_scatter(axes[0,0], x_raw, y_raw, "Raw Data (Base)", "yellow")
    # Add WLS line to Raw plot
    # Create a range of x values for plotting line
    x_range = np.linspace(x_raw.min(), x_raw.max(), 100)
    X_range = sm.add_constant(x_range)
    y_wls = model_wls.predict(X_range)
    axes[0,0].plot(x_range, y_wls, color='red', linestyle='--', linewidth=2, label=r'WLS Fit (w=1/$\sigma^2$)')
    axes[0,0].legend()
    axes[0,0].set_xlabel("Theta (t)")
    axes[0,0].set_ylabel("Theta (t+1d)")
    
    # 2. Log Transform
    plot_scatter(axes[0,1], x_log, y_log, "Log Transform (log(x))", "cyan")
    axes[0,1].set_xlabel("log(Theta t)")
    axes[0,1].set_ylabel("log(Theta t+1d)")
    
    # 3. Sqrt Transform
    plot_scatter(axes[1,0], x_sqrt, y_sqrt, "Sqrt Transform (sqrt(x))", "lime")
    axes[1,0].set_xlabel("sqrt(Theta t)")
    axes[1,0].set_ylabel("sqrt(Theta t+1d)")
    
    # 4. Box-Cox Transform
    plot_scatter(axes[1,1], x_boxcox, y_boxcox, f"Box-Cox Transform (\u03bb={lmbda:.2f})", "magenta")
    axes[1,1].set_xlabel("Box-Cox(Theta t)")
    axes[1,1].set_ylabel("Box-Cox(Theta t+1d)")
    
    plt.suptitle("Variance Stabilization for Theta Predictability", fontsize=16)
    plt.tight_layout()
    
    out_dir = "returns_research/deliverables/04_hurst_predictability"
    plt.savefig(f"{out_dir}/theta_stabilization.png", dpi=300, bbox_inches='tight')
    print("Saved theta_stabilization.png")
    plt.close()

if __name__ == "__main__":
    main()
