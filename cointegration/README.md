# 📊 Cointegration Analysis Module

This module contains all components related to **cointegration discovery**, **persistence testing**, and **relationship analysis** for currency pairs in the FX market.

## 🏗 Structure

### 📁 `scripts/`
Core mathematical engines for cointegration analysis:
- `cointegration_analysis.py`: GPU-accelerated universe scanner that tests all pair combinations
- `cointegration_persistence_v2.py`: Multi-stream optimizer that measures relationship stability over time

### 📁 `notebooks/`
Interactive research environment:
- `cointegration_analysis.ipynb`: Visual exploration of cointegration matrices and relationships
- `visualize_hedge_ratio.ipynb`: Deep dive into specific pairs and spread dynamics

### 📁 `persistence_analysis/`
Alpha decay research:
- `generate_plots.py`: Statistical visualization tool for persistence metrics
- `plots/`: Output directory for generated charts

### 📁 `results/`
Analysis outputs:
- `daily_cointegration_results_gpu.pkl`: Daily cointegration scores for all pairs
- `persistence_v2_fixed_beta.pkl`: Persistence analysis results across multiple timeframes

## 🚀 Quick Start

### 1. Run Cointegration Analysis
```bash
cd scripts
python cointegration_analysis.py
```

This scans all currency pairs in `../data/` and generates cointegration scores.

### 2. Test Persistence
```bash
python cointegration_persistence_v2.py
```

Analyzes how stable cointegration relationships are over time (14, 30, 90 day windows).

### 3. Generate Visualizations
```bash
cd ../persistence_analysis
python generate_plots.py
```

Creates charts showing persistence rates, beta stability, and score degradation.

## 📊 Key Concepts

**Cointegration**: Two price series that share a common stochastic trend, ensuring their spread is mean-reverting.

**ADF Score**: Augmented Dickey-Fuller t-statistic. More negative = stronger mean reversion.

**Beta (β)**: The hedge ratio. For pair (Y, X), the spread is $S = Y - \beta X$.

**Persistence**: The probability that a cointegrated pair today remains cointegrated tomorrow.

## ⚡ Performance

All scripts use GPU acceleration via RAPIDS (cuDF) and CuPy for:
- Vectorized OLS regression across all pairs simultaneously
- Parallel ADF testing
- Multi-stream computation for different time windows

Expected runtime on typical GPU: **~2-5 minutes** for full universe analysis.
