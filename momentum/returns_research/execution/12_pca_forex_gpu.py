import sys
import os
import cudf
import cupy as cp
import numpy as np
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
    print(f"Found {len(symbols)} symbols: {symbols}")
    
    dfs = []
    for sym in symbols:
        print(f"Loading {sym}...")
        df = data_feed.load_data(sym, timeframe='1min')
        if df is not None:
            # Calculate Log Returns
            # We use Close price
            # df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
            # CuDF doesn't support np.log on series directly in some versions, use cupy or apply
            # But Series.diff() on log prices is safer/easier
            df['log_price'] = np.log(df['Close'])
            df['ret'] = df['log_price'].diff()
            
            # Keep only returns, rename col to symbol
            series = df['ret'].dropna()
            series.name = sym
            dfs.append(series)
            
    print("Aligning data (this may take a moment)...")
    # Concatenate along columns (axis=1) - outer join by default or aligned by index?
    # cudf.concat with axis=1 aligns by index.
    # We want INNER join to have valid data for all.
    
    # Efficient merge: start with first, join others. 
    # Or just concat and dropna.
    if not dfs: return None
    
    wide_df = cudf.concat(dfs, axis=1)
    wide_df.columns = [s.name for s in dfs]
    
    # Drop rows with any NaN (inner join equivalent for valid returns)
    wide_df = wide_df.dropna()
    print(f"Aligned Data Shape: {wide_df.shape}")
    
    return wide_df

def main():
    # 1. Load Data
    returns_df = get_aligned_returns()
    if returns_df is None or len(returns_df) == 0:
        print("No aligned data found.")
        return

    # 2. Static PCA (Global Structure)
    print("Computing Static PCA...")
    
    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(returns_df)
    
    # PCA
    n_components = min(len(returns_df.columns), 10)
    pca = cuPCA(n_components=n_components)
    pca.fit(X_scaled)
    
    explained_var = pca.explained_variance_ratio_.to_pandas()
    components = pca.components_.to_pandas()
    feature_names = returns_df.columns
    
    # --- PLOT 1: Explained Variance ---
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(1, len(explained_var)+1), explained_var * 100, color='#00CED1', alpha=0.8)
    plt.plot(range(1, len(explained_var)+1), np.cumsum(explained_var)*100, color='#FFD700', marker='o', label='Cumulative')
    
    plt.title('PCA Explained Variance: How many factors drive the Forex Market?', fontsize=16)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', color='white')
                 
    out_dir = "returns_research/deliverables/05_pca_analysis"
    plt.savefig(f"{out_dir}/01_explained_variance.png", dpi=300, bbox_inches='tight')
    print("Saved 01_explained_variance.png")
    plt.close()
    
    # --- PLOT 2: Loadings Map (PC1 vs PC2) ---
    # Visualize correlations.
    # PC1 usually USD strength. PC2 usually Risk (JPY/CHF vs AUD/NZD) or Euro strength.
    
    plt.figure(figsize=(14, 10))
    
    # Components shape: (n_components, n_features)
    # PC1 loadings: components.iloc[0, :]
    # PC2 loadings: components.iloc[1, :]
    
    pc1_loadings = components.iloc[0, :].values
    pc2_loadings = components.iloc[1, :].values
    
    plt.scatter(pc1_loadings, pc2_loadings, c='#FF00FF', s=100, alpha=0.8, edgecolors='white')
    
    for i, txt in enumerate(feature_names):
        plt.annotate(txt, (pc1_loadings[i], pc2_loadings[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=12, color='white')
                     
    plt.axhline(0, color='white', linestyle='--', alpha=0.3)
    plt.axvline(0, color='white', linestyle='--', alpha=0.3)
    plt.title('Forex Market Map (PC1 vs PC2 Loadings)', fontsize=16)
    plt.xlabel(f'PC1 Loadings ({explained_var[0]*100:.1f}% Var) - Usually Global USD Factor', fontsize=12)
    plt.ylabel(f'PC2 Loadings ({explained_var[1]*100:.1f}% Var) - Usually Risk/Regional Factor', fontsize=12)
    plt.grid(True, alpha=0.2)
    
    plt.savefig(f"{out_dir}/02_loadings_map.png", dpi=300, bbox_inches='tight')
    print("Saved 02_loadings_map.png")
    plt.close()
    
    # 3. Rolling PCA Analysis (Dynamics)
    print("Computing Rolling PCA (4-Day Window)...")
    
    # We'll calculate the Explained Variance of PC1 over time
    # This indicates "Market Synchronization".
    # High PC1 Var = Everything moves together (Crisis/Dollar Spike).
    # Low PC1 Var = Idiosyncratic moves (Pairs doing their own thing).
    
    window = 4 * 24 * 60 # 4 Days
    
    # Rolling Covariance Matrix is hard on memory for long history in one go.
    # Optimized approach: Calculate Rolling Correlation Sums? 
    # Actually, approximations or just calculating every N steps is better for viz.
    # Let's stride: Calculate PCA every 60 minutes (1 hour) on the last 4 days window.
    
    # Convert to pandas/numpy for stride iteration (easier loop, still fast enough if window calc is GPU)
    # Actually, we can just grab chunks.
    
    timestamps = returns_df.index.to_pandas()
    # Stride of 60 mins
    stride = 60
    
    dates = []
    pc1_vars = []
    
    # We iterate. For GPU speed, we could assume strict rolling, but cuPCA doesn't support "rolling" directly.
    # We will slice indices.
    
    total_len = len(returns_df)
    
    # Pre-compute X_scaled is tricky because rolling standardization is needed strictly speaking.
    # But global standardization is okay for "structure" approximation.
    # Let's do Rolling Z-Score ideally?
    # For speed/simplicity of this specific request, we slice the global scaled data? 
    # No, local standardization is better for PCA.
    
    # Convert entire df to cupy for fast slicing
    data_matrix = returns_df.values
    
    # Loop
    # Start from window size
    indices = range(window, total_len, stride)
    
    # Using a list to collect results
    pc1_explained = []
    plot_dates = []
    
    print(f"Processing {len(indices)} windows...")
    
    for i in indices:
        # Slice [i-window : i]
        window_data = data_matrix[i-window : i]
        
        # Fit PCA on this chunk
        # Small PCA (N=window, Feats=N_sym) is very fast on GPU
        local_pca = cuPCA(n_components=1)
        local_pca.fit(window_data)
        
        # explained_variance_ratio_ is likely a cupy array.
        # We extract the first element and ensure it's a python float/numpy scalar
        val = local_pca.explained_variance_ratio_
        if hasattr(val, 'to_pandas'):
            val = val.to_pandas()[0]
        elif hasattr(val, 'get'): # cupy
            val = val.get()[0]
        else: # numpy
            val = val[0]
            
        pc1_explained.append(val)
        plot_dates.append(timestamps[i])
        
    # Plot Rolling Synchronization
    plt.figure(figsize=(16, 6))
    plt.plot(plot_dates, pc1_explained, color='#00FF7F', linewidth=1)
    
    plt.title(f'Market Synchronization: Rolling Explained Variance of PC1 (4-Day Window)', fontsize=16)
    plt.ylabel('Variance Explained by PC1')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.2)
    
    plt.savefig(f"{out_dir}/03_rolling_synchronization.png", dpi=300, bbox_inches='tight')
    print("Saved 03_rolling_synchronization.png")
    plt.close()
    
    # 4. Predictability: Does Market Sync predict Returns?
    # "How they influence each other" -> If Sync is High, does Mean Reversion fail?
    # Let's save the Sync data to csv for potential future use or correlation check
    
    # For now, the visual is the deliverables.

if __name__ == "__main__":
    main()
