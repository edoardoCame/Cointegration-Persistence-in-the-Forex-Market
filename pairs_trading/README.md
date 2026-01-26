# 📉 Pairs Trading: Walk-Forward Backtester

This directory contains the execution logic that simulates the strategy on historical data. It is designed to be **rigorous** and **production-ready**.

## 🧠 Core Philosophy: Walk-Forward Validation
Unlike standard backtests that fit parameters over the whole period (Lookahead Bias), this system simulates a trader waking up each morning:
1.  **Morning (Training)**: Look at the *past* 90 days. Calculate Cointegration and Beta. Select the Top 5 pairs.
2.  **Intraday (Testing)**: Trade *today* using those fixed parameters.
3.  **Repeat**: Move to tomorrow.

## 📁 Key Files

### `run_backtest.py`
The main simulation engine.
*   **Data Loading**: Loads M1 data into GPU memory.
*   **Calibration**: Computes OLS Beta and Mean/Std of spread on the fly (GPU).
*   **Simulation**: Uses `numba.jit` to run an event-driven loop bar-by-bar.
    *   *1-Bar Lag*: Signals calculated at Close[t] are executed at Close[t+1] (simulating next-bar execution) to guarantee no peeking.
*   **Outputs**:
    *   `detailed_trades.csv`: Every single trade execution.
    *   `daily_betas.csv`: Evolution of hedge ratios.
    *   `backtest_results.csv`: Daily aggregated PnL.

### `generate_plots.py`
The visualization suite.
*   **Equity Curve**: Cumulative PnL & Drawdown.
*   **Beta Evolution**: Tracks how the Hedge Ratio changes over time for specific pairs.
*   **Positions Timeline**: A Gantt-style chart showing market exposure.
*   **Trade Analysis**: Histograms of PnL and trade duration.

## ⚙️ Strategy Logic
*   **Entry**: Z-Score > 1.0 (Short Spread) or < -1.0 (Long Spread).
*   **Exit**: Z-Score crosses 0.0 (Mean Reversion).
*   **Sizing**: Fixed capital allocation per pair.
