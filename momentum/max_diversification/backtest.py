import sys
import os
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cuml import PCA as cuPCA
from cuml.preprocessing import StandardScaler

# Add project root
sys.path.append(os.getcwd())
from returns_research.system import data_feed

def get_aligned_data_daily():
    """Daily data is sufficient and cleaner for Portfolio Optimization."""
    symbols = data_feed.get_all_symbols()
    dfs = []
    for sym in symbols:
        df = data_feed.load_data(sym, timeframe='1D')
        if df is not None:
            series = df['Close']
            series.name = sym
            dfs.append(series)
    if not dfs: return None
    return cudf.concat(dfs, axis=1).dropna()

def backtest_max_diversification():
    print("Loading Data (Daily)...")
    prices = get_aligned_data_daily()
    if prices is None: return

    returns = np.log(prices).diff().dropna()
    
    # Configuration
    LOOKBACK_MONTHS = 3
    REBALANCE_FREQ = '1M'
    
    dates = returns.index.to_pandas()
    rebalance_dates = pd.Series(index=dates, data=1).resample(REBALANCE_FREQ).last().index
    
    portfolio_returns = []
    
    print(f"Starting PCA Risk Parity Backtest...")
    
    for i in range(len(rebalance_dates)-1):
        train_end = rebalance_dates[i]
        trade_end = rebalance_dates[i+1]
        
        # 1. Training (Lookback 3 months)
        train_start = train_end - pd.DateOffset(months=LOOKBACK_MONTHS)
        train_mask = (returns.index > train_start) & (returns.index <= train_end)
        train_data = returns.loc[train_mask]
        
        if len(train_data) < 20: continue # Min data for cov
        
        # PCA to find risk structure
        # Standardize for factor analysis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_data)
        
        n_comp = 5 # Balance across top 5 factors
        pca = cuPCA(n_components=n_comp)
        pca.fit(X_scaled)
        
        # PCA Risk Parity / Diversification logic:
        # We want to allocate weights w such that contribution to risk from PC factors is equal.
        # A simple robust proxy for Max Diversification using PCA is to weight assets 
        # by 1 / (Sum of absolute loadings on top PCs).
        # This penalizes assets that are heavily exposed to the main systemic factors.
        
        loadings = pca.components_ # (n_comp, n_features)
        # Weight per asset: 1 / Sum(abs(loadings))
        systemic_exposure = cp.sum(cp.abs(loadings.values), axis=0)
        
        # weights = 1 / systemic_exposure
        raw_weights = 1.0 / (systemic_exposure + 1e-6)
        
        # Normalize weights to sum to 1
        weights = raw_weights / cp.sum(raw_weights)
        weights_host = cp.asnumpy(weights)
        
        # 2. Trading (Next Month)
        trade_mask = (returns.index > train_end) & (returns.index <= trade_end)
        trade_data = returns.loc[trade_mask].to_pandas()
        
        if len(trade_data) == 0: continue
        
        # Portfolio Daily Returns
        period_returns = trade_data.dot(weights_host)
        portfolio_returns.extend(period_returns.values)
        
    # --- Performance ---
    pnl = np.array(portfolio_returns)
    cum_pnl = np.cumsum(pnl)
    
    # Compare with Equal Weight (1/N)
    # Simple benchmark logic inside the same loop would be better, but let's just plot result.
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    plt.plot(cum_pnl, color='#ADFF2F', label='PCA Max-Diversification')
    plt.title('Strategy 3: PCA-Based Max Diversification Portfolio', fontsize=16)
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.2)
    
    out_dir = "returns_research/strategies/max_diversification"
    plt.savefig(f"{out_dir}/backtest_performance.png", dpi=300, bbox_inches='tight')
    print("Saved backtest_performance.png")

if __name__ == "__main__":
    backtest_max_diversification()
