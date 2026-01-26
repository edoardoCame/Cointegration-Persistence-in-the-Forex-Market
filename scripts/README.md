# 🧮 Scripts: The Math Engine

This directory contains the heavy-lifting scripts responsible for **Universe Selection** and **Cointegration Discovery**.

## Core Components

### 1. `cointegration_analysis.py`
**The Universe Scanner.**
*   **Function**: Loads all available currency pairs and computes the Engle-Granger cointegration test for *every possible combination* ($N 	imes N$ matrix).
*   **Methodology**:
    *   Uses **OLS (Ordinary Least Squares)** to determine the Hedge Ratio ($eta$).
    *   Calculates the **ADF (Augmented Dickey-Fuller)** t-statistic on the residuals.
    *   Runs on a rolling 7-day window.
*   **Why GPU?**: Computing OLS and ADF for hundreds of pairs over millions of data points involves massive matrix operations. The GPU reduces this from hours to seconds.

### 2. `cointegration_persistence_v2.py`
**The Alpha Decay Tester.**
*   **Function**: Analyzes how "persistent" a cointegration relationship is.
*   **Logic**:
    *   trains on window $W$ (e.g., 90 days).
    *   Tests on window $T+1$ (the next day).
    *   Compares the "In-Sample" score vs. "Out-of-Sample" score.
*   **Performance**: Uses **CUDA Streams** to calculate multiple timeframes (14, 30, 90 days) in parallel on the GPU, minimizing synchronization overhead.

## 📐 Mathematical Note
We utilize **Vectorized OLS**. Instead of looping through pairs, we solve the regression equation for the entire matrix simultaneously:

$$ \beta_{ij} = \frac{Cov(Y_i, X_j)}{Var(X_j)} $$

This approach ensures strict mathematical correctness while maximizing hardware utilization.
