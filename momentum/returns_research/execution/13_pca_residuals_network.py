import sys
import os
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cuml import PCA as cuPCA
from cuml.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.getcwd())

from returns_research.system import data_feed

def get_aligned_returns():
    """Loads all symbols and aligns them into a single wide DataFrame."""
    symbols = data_feed.get_all_symbols()
    dfs = []
    for sym in symbols:
        df = data_feed.load_data(sym, timeframe='1min')
        if df is not None:
            df['log_price'] = np.log(df['Close'])
            df['ret'] = df['log_price'].diff()
            series = df['ret'].dropna()
            series.name = sym
            dfs.append(series)
    
    if not dfs: return None
    wide_df = cudf.concat(dfs, axis=1)
    wide_df.columns = [s.name for s in dfs]
    return wide_df.dropna()

def main():
    print("Loading Data for Network Analysis...")
    returns_df = get_aligned_returns()
    if returns_df is None: return

    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(returns_df)
    
    # 1. Fit PCA (1 Component - The "Global Factor")
    pca = cuPCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)
    
    # Convert to cupy for math
    # X_pca might be cuDF or cupy. cuml 23.12 returns input type usually.
    if hasattr(X_pca, 'values'):
        X_pca_vals = X_pca.values
    else:
        X_pca_vals = X_pca
        
    if hasattr(pca.components_, 'values'):
        comps_vals = pca.components_.values
    else:
        comps_vals = pca.components_
        
    if hasattr(X_scaled, 'values'):
        X_scaled_vals = X_scaled.values
    else:
        X_scaled_vals = X_scaled

    print(f"Shapes - X_pca: {X_pca_vals.shape}, Components: {comps_vals.shape}")
    
    # Ensure X_pca is 2D (N, 1)
    if X_pca_vals.ndim == 1:
        X_pca_vals = X_pca_vals.reshape(-1, 1)

    # Reconstruct the "Market Portion" of returns
    # X_recon = T * W 
    X_recon = cp.dot(X_pca_vals, comps_vals)
    
    # 2. Calculate "Systemicness" (R2 with PC1)
    # Residuals = X_scaled - X_recon
    residuals = X_scaled_vals - X_recon
    
    # Calculate R2 for each column (pair)
    # SS_tot = sum(X_scaled^2) -> which is N (since std=1)
    # SS_res = sum(residuals^2)
    
    ss_tot = cp.sum(X_scaled_vals**2, axis=0)
    ss_res = cp.sum(residuals**2, axis=0)
    r2_systemic = 1 - (ss_res / ss_tot)
    
    # Convert to pandas for plotting
    pairs = returns_df.columns
    r2_series = pd.Series(cp.asnumpy(r2_systemic), index=pairs).sort_values(ascending=False)
    
    # --- PLOT 1: Who is the Slave of the Market? ---
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 8))
    
    # Color map
    norm = plt.Normalize(r2_series.min(), r2_series.max())
    colors = plt.cm.plasma(norm(r2_series.values))
    
    bars = plt.bar(r2_series.index, r2_series.values * 100, color=colors)
    
    plt.title('Systemicness: % of Variance Explained by Global Factor (PC1)', fontsize=16)
    plt.ylabel('% Variance from Global Factor')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.2)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f}%', ha='center', va='bottom', fontsize=9, color='white')

    out_dir = "returns_research/deliverables/05_pca_analysis"
    plt.savefig(f"{out_dir}/04_systemic_exposure.png", dpi=300, bbox_inches='tight')
    print("Saved 04_systemic_exposure.png")
    plt.close()
    
    # --- PLOT 2: The Hidden Network (Residual Correlations) ---
    # Now we correlate the RESIDUALS.
    # This removes the "Rising tide lifts all boats" effect.
    # It shows who is swimming together underwater.
    
    # Calculate Correlation Matrix of Residuals
    # residuals is (N_samples, N_features) cupy array
    
    # Covariance of residuals
    # Since residuals mean is approx 0
    res_corr = cp.corrcoef(residuals, rowvar=False)
    res_corr_host = cp.asnumpy(res_corr)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(res_corr_host, xticklabels=pairs, yticklabels=pairs, 
                cmap='vlag', center=0, vmin=-0.8, vmax=0.8,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
                
    plt.title('The Hidden Network: Correlation of Idiosyncratic Residuals (Global Factor Removed)', fontsize=16)
    plt.savefig(f"{out_dir}/05_residual_network.png", dpi=300, bbox_inches='tight')
    print("Saved 05_residual_network.png")
    plt.close()

if __name__ == "__main__":
    main()
