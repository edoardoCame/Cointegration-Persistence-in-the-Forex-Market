# 🚀 DARWINEX Mission: GPU-Accelerated Statistical Arbitrage

## Overview
This repository hosts a high-performance **Statistical Arbitrage (Pairs Trading)** engine designed for the FX market. It leverages **RAPIDS (cuDF)** and **CuPy** to perform heavy statistical computations on the GPU, enabling the analysis of massive high-frequency datasets (M1 bars) in seconds rather than hours.

## 🏗 System Architecture

The system is divided into three logical layers:

1.  **Discovery Layer (`cointegration/scripts/`)**: Scans the entire universe of currency pairs to find cointegrated relationships using OLS (Ordinary Least Squares) and ADF tests.
2.  **Validation Layer (`cointegration/persistence_analysis/`)**: Tests the durability of these relationships. Does a pair that looks good today stay good tomorrow?
3.  **Execution Layer (`BB backtest/`)**: A rigorous Walk-Forward backtester that simulates the actual trading strategy with zero lookahead bias.

## ⚡ Key Technologies

*   **GPU Acceleration**: All vector algebra (Covariance matrices, Eigenvalues, OLS regression) happens on NVIDIA GPUs via `cupy` and `cudf`.
*   **JIT Compilation**: The event-driven backtesting loop is compiled to machine code using `numba` for C++ level performance in Python.
*   **Math**: Switched from TLS (Total Least Squares) to **OLS (Ordinary Least Squares)** to ensure standard interpretation of Hedge Ratios ($\beta$).

## 📂 Directory Structure

*   `data/`: Raw M1 OHLCV data.
*   `cointegration/`: Cointegration analysis and research.
    *   `scripts/`: Core math engines for cointegration discovery.
    *   `persistence_analysis/`: Research on alpha decay.
    *   `notebooks/`: Interactive visualizations and prototyping.
    *   `results/`: Analysis outputs, pickled models.
*   `BB backtest/`: Backtesting framework and validation.