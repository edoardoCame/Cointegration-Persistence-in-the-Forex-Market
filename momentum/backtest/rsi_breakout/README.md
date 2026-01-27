# RSI Breakout Backtesting System

GPU-accelerated RSI breakout strategy backtesting using **Walk Forward Optimization (WFO)**, **Random Portfolio**, and **Monte Carlo** simulation.

## Project Structure

```
rsi_breakout/
├── 01_rsi_wfo_gpu.py           # Walk Forward Optimization with GPU acceleration
├── 02_rsi_portfolio_random.py   # Random portfolio with weekly rebalancing
├── 03_rsi_monte_carlo.py        # 1000x Monte Carlo simulations
├── results/
│   ├── wfo/                     # WFO optimization results
│   │   ├── equity_curve.csv
│   │   ├── equity_curve.png
│   │   ├── param_history.csv
│   │   └── params.png
│   ├── portfolio/               # Random portfolio backtest results
│   │   ├── portfolio_equity.csv
│   │   └── portfolio_equity.png
│   └── monte_carlo/             # Monte Carlo distribution results
│       ├── monte_carlo_results.csv
│       └── monte_carlo_distribution.png
└── README.md
```

## Strategy Configuration

- **Asset**: GBPJPY (minute-level data)
- **Indicator**: RSI (Wilder's smoothing)
- **Parameters**:
  - Lookback: 60-30,000 minutes
  - Upper threshold: 55-95
  - Lower threshold: 5-45
- **Position**: Always-in (long/short/flat based on RSI levels)
- **Costs**: 1.2 pips per round-trip (JPY pair adjustment)

## Running the Backtests

### 1. Walk Forward Optimization (WFO)
```bash
python 01_rsi_wfo_gpu.py
```
- 90-day in-sample optimization window
- 7-day out-of-sample testing window
- Weekly parameter re-selection
- GPU-accelerated RSI calculation & parameter optimization

**Output**: `results/wfo/`

### 2. Random Portfolio Backtest
```bash
python 02_rsi_portfolio_random.py
```
- 10 random RSI parameter combinations
- Weekly equal-weight rebalancing
- Full historical backtest

**Output**: `results/portfolio/`

### 3. Monte Carlo Analysis (1000 runs)
```bash
python 03_rsi_monte_carlo.py
```
- Pre-calculates all 500 RSI lookbacks on GPU (~0.6s)
- Runs 1000 portfolio simulations with random parameters
- Generates distribution statistics and histogram

**Output**: `results/monte_carlo/`

## Implementation Details

### GPU Acceleration (Numba CUDA)
- **RSI Kernel**: Calculates Wilder's smoothed RSI for 500 lookbacks in parallel
- **WFO Kernel**: Tests 38,637+ parameter combinations per optimization window

### Timing Correctness
All implementations follow **correct lookahead-free logic**:
1. Calculate returns using position from previous bar
2. Read signal at NEXT bar to determine new position
3. Deduct transaction costs on position changes

This prevents the "future sight" problem common in naive backtests.

## Requirements

- Python 3.10+
- RAPIDS 23.12 (cuDF, CuPy)
- Numba with CUDA support
- Pandas, NumPy, Matplotlib

## Performance

- RSI pre-calculation: ~0.58s (500 lookbacks)
- WFO optimization: ~50s (full dataset)
- Portfolio simulation: ~1.3 seconds per run
- Monte Carlo (1000 runs): ~2-3 minutes total
