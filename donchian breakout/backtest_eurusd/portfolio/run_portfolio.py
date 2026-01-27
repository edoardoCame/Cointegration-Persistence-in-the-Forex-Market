
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPONENTS_DIR = os.path.join(BASE_DIR, "components")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Ensure output dir exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_components():
    print(f"Loading components from {COMPONENTS_DIR}...")
    files = glob.glob(os.path.join(COMPONENTS_DIR, "*.csv"))
    if not files:
        print("No component files found!")
        return None
        
    dfs = []
    
    for f in files:
        fname = os.path.basename(f)
        prefix = fname.split('.')[0].upper() # VOL, RSI
        print(f"  Reading {fname}...")
        
        # Index is DateTime, parse it
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        
        # Rename columns to ensure uniqueness
        df.columns = [f"{prefix}_{c}" for c in df.columns]
        dfs.append(df)
        
    print("Aligning components...")
    # Concat along columns (axis=1)
    full_df = pd.concat(dfs, axis=1)
    
    # Forward fill gaps if any, fillna with 0 (assuming gaps are inactive)
    # actually PnL curves should be 0 before start.
    full_df = full_df.ffill().fillna(0.0)
    
    return full_df

def run_portfolio():
    equity_pnl = load_components()
    if equity_pnl is None:
        return

    print(f"Total Components: {len(equity_pnl.columns)}")
    print(f"Date Range: {equity_pnl.index.min()} to {equity_pnl.index.max()}")
    
    # --- Convert to Returns for Portfolio Simulation ---
    # The 'Equities' loaded are Cumulative PnL curves in Price Units (e.g. 0.50 means 50 cents move captured).
    # To convert this to % Returns, we approximate the asset price (EURUSD ~ 1.10).
    # Return = PnL_Delta / Asset_Price
    
    print("Calculating Returns (assuming 1:1 Leverage and Avg Price 1.10)...")
    pnl_delta = equity_pnl.diff().fillna(0.0)
    
    # Approx Asset Price
    ASSET_PRICE_AVG = 1.10
    
    # Unleveraged Returns (1:1)
    returns_df = pnl_delta / ASSET_PRICE_AVG
    
    # Reconstruct a Notional Index for Volatility Calculation
    # start at 1.0
    price_series = (1 + returns_df).cumprod()
    
    # --- WEEKLY REBALANCING LOGIC ---
    
    # 1. Calculate Volatility Estimator (Risk)
    # Use Rolling 30-Day Standard Deviation of Returns
    
    print("Calculating Volatility metrics...")
    daily_prices = price_series.resample('D').last().ffill()
    daily_rets = daily_prices.pct_change().fillna(0.0)
    
    # Rolling 20-Day Volatility (approx 1 trading month)
    rolling_vol = daily_rets.rolling(window=20, min_periods=5).std()
    
    # Replace 0 vol with mean (to avoid div by zero)
    rolling_vol = rolling_vol.replace(0.0, rolling_vol.mean())
    
    # 2. Determine Weights at Week Start
    # Rebalance Frequency: Weekly (Fridays close / Monday open)
    # We take the vol measure at Friday close, calculate weights, apply to next week.
    
    print("Calculating Weights...")
    weekly_vol = rolling_vol.resample('W-FRI').last()
    
    # -- Allocation A: Equal Weight --
    # Weight = 1/N
    n_strats = len(returns_df.columns)
    w_ew = pd.DataFrame(1.0 / n_strats, index=weekly_vol.index, columns=weekly_vol.columns)
    
    # -- Allocation B: Risk Parity --
    # Weight ~ 1 / Vol
    inv_vol = 1.0 / weekly_vol
    # Normalize row-wise so sum(weights) = 1.0 (Full investment)
    w_rp = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    
    # Handle NaNs (early period) -> Default to Equal Weight
    w_rp = w_rp.fillna(1.0 / n_strats)
    
    # 3. Apply Weights (Resulting in Portfolio Returns)
    # Shift weights forward 1 period (Week T's weights apply to Week T+1)
    w_ew_shifted = w_ew.shift(1).fillna(1.0 / n_strats)
    w_rp_shifted = w_rp.shift(1).fillna(1.0 / n_strats)
    
    # Upsample to Minute resolution (ffill) to match returns_df
    # Align dates
    w_ew_m1 = w_ew_shifted.reindex(returns_df.index).ffill().fillna(1.0 / n_strats)
    w_rp_m1 = w_rp_shifted.reindex(returns_df.index).ffill().fillna(1.0 / n_strats)
    
    print("Simulating Portfolio Performance...")
    # Portfolio Return = Sum (Weight_i * Return_i)
    port_ret_ew = (returns_df * w_ew_m1).sum(axis=1)
    port_ret_rp = (returns_df * w_rp_m1).sum(axis=1)
    
    # Calculate Cumulative Wealth (Equity Curve)
    # Start with $100,000 Portfolio
    initial_capital = 100000.0
    equity_ew = initial_capital * (1 + port_ret_ew).cumprod()
    equity_rp = initial_capital * (1 + port_ret_rp).cumprod()
    
    # --- OUTPUTS ---
    
    # 1. Stats
    total_ret_ew = (equity_ew.iloc[-1] / initial_capital) - 1
    total_ret_rp = (equity_rp.iloc[-1] / initial_capital) - 1
    
    # Approx 8 years
    years = (equity_ew.index[-1] - equity_ew.index[0]).days / 365.25
    cagr_ew = (equity_ew.iloc[-1] / initial_capital) ** (1/years) - 1
    cagr_rp = (equity_rp.iloc[-1] / initial_capital) ** (1/years) - 1
    
    print("\n--- Results ---")
    print(f"Years: {years:.2f}")
    if years > 0:
        print(f"Equal Weight | Total: {total_ret_ew*100:.2f}% | CAGR: {cagr_ew*100:.2f}% | Final: ${equity_ew.iloc[-1]:,.2f}")
        print(f"Risk Parity  | Total: {total_ret_rp*100:.2f}% | CAGR: {cagr_rp*100:.2f}% | Final: ${equity_rp.iloc[-1]:,.2f}")
    
    # 2. CSV Export
    out_df = pd.DataFrame({
        "Equal_Weight": equity_ew,
        "Risk_Parity": equity_rp
    })
    csv_path = os.path.join(OUTPUT_DIR, "portfolio_equity.csv")
    out_df.to_csv(csv_path)
    print(f"Saved equity curves to {csv_path}")
    
    # 3. Plot
    plt.figure(figsize=(12, 8))
    plt.plot(equity_ew, label=f"Equal Weight (CAGR {cagr_ew*100:.1f}%)", linewidth=1.5)
    plt.plot(equity_rp, label=f"Risk Parity (CAGR {cagr_rp*100:.1f}%)", linewidth=1.5)
    plt.title("Portfolio Strategy Comparison (Weekly Rebalance)")
    plt.ylabel("Portfolio Equity ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "portfolio_comparison.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    # Also plot the weights over time to see allocation shifts
    plt.figure(figsize=(12, 6))
    plt.stackplot(w_rp_shifted.index, w_rp_shifted.T, labels=w_rp_shifted.columns, alpha=0.6)
    plt.title("Risk Parity Weights Allocation Over Time")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "risk_parity_weights.png"))

if __name__ == "__main__":
    run_portfolio()
