import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = Path("pairs_trading/backtest_results.csv")
if not results_file.exists():
    print("Error: Results file not found.")
    exit(1)

df = pd.read_csv(results_file, parse_dates=['date'], index_col='date')

# Plot Cumulative PnL
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['cum_pnl'], label='Cumulative PnL', color='blue', linewidth=2)
plt.fill_between(df.index, df['cum_pnl'], alpha=0.1, color='blue')

# Styling
plt.title('Pairs Trading Backtest Equity Curve (Dec 2025)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('PnL ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save plot
output_file = Path("pairs_trading/equity_curve.png")
plt.savefig(output_file)
print(f"Equity curve saved to {output_file}")
