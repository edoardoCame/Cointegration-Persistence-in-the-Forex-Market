# 🔍 Persistence Analysis

This directory is dedicated to researching the **decay characteristics** of our alpha. In mean-reversion trading, a key risk is that a pair is cointegrated *in-sample* (past) but diverges *out-of-sample* (future).

## 🎯 Objective
To quantify the probability that a pair with a strong ADF score today will continue to revert to the mean tomorrow.

## 🛠 Tools

### `generate_plots.py`
Generates statistical visualizations from the pickle files created by `scripts/cointegration_persistence_v2.py`.
*   **Beta Stability**: Scatter plots showing $\beta_{t}$ vs $\beta_{t+1}$.
*   **Score Degradation**: How much does the ADF score worsen out-of-sample?
*   **Win Rate by Decile**: Does a stronger ADF score actually predict a higher probability of profit?

## 📊 Key Metrics
*   **Persistence Rate**: The % of pairs that remain significant (p < 0.05) in the subsequent period.
*   **Half-Life**: The estimated time it takes for a spread deviation to revert by half (calculated via Ornstein-Uhlenbeck process).
