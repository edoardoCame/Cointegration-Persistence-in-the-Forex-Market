
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import numpy as np

# Config
RESULTS_DIR = Path("pairs_trading")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)

def load_data():
    try:
        daily = pd.read_csv(RESULTS_DIR / "backtest_results.csv", parse_dates=['date'], index_col='date')
        trades = pd.read_csv(RESULTS_DIR / "detailed_trades.csv", parse_dates=['entry_time', 'exit_time'])
        betas = pd.read_csv(RESULTS_DIR / "daily_betas.csv", parse_dates=['date'])
        return daily, trades, betas
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Run backtest first.")
        return None, None, None

def plot_equity_curve(daily):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity
    ax1.plot(daily.index, daily['cum_pnl'], color='#1f77b4', linewidth=2, label='Cumulative PnL')
    ax1.fill_between(daily.index, daily['cum_pnl'], color='#1f77b4', alpha=0.1)
    ax1.set_title("Portfolio Equity Curve", fontsize=16, fontweight='bold')
    ax1.set_ylabel("PnL ($)")
    ax1.legend(loc='upper left')
    
    # Drawdown
    running_max = daily['cum_pnl'].cummax()
    drawdown = daily['cum_pnl'] - running_max
    
    ax2.fill_between(daily.index, drawdown, 0, color='#d62728', alpha=0.3, label='Drawdown')
    ax2.plot(daily.index, drawdown, color='#d62728', linewidth=1)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_equity_curve.png", dpi=300)
    print("Saved 1_equity_curve.png")
    plt.close()

def plot_trade_analysis(trades):
    if trades.empty: return
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. PnL Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(trades['pnl'], kde=True, ax=ax1, color='green', bins=50)
    ax1.axvline(0, color='black', linestyle='--')
    ax1.set_title("Trade PnL Distribution")
    ax1.set_xlabel("PnL ($)")
    
    # 2. PnL by Hour of Day
    ax2 = fig.add_subplot(gs[0, 1])
    trades['hour'] = trades['entry_time'].dt.hour
    hourly_pnl = trades.groupby('hour')['pnl'].sum()
    sns.barplot(x=hourly_pnl.index, y=hourly_pnl.values, ax=ax2, palette="viridis")
    ax2.set_title("Total PnL by Hour of Day")
    ax2.set_ylabel("Total PnL ($)")
    
    # 3. Scatter: Duration vs PnL
    ax3 = fig.add_subplot(gs[1, :])
    trades['duration_min'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 60
    
    sns.scatterplot(data=trades, x='duration_min', y='pnl', hue='direction', alpha=0.6, ax=ax3)
    ax3.set_title("Trade Duration vs PnL")
    ax3.set_xlabel("Duration (Minutes)")
    ax3.set_ylabel("PnL ($)")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "2_trade_analysis.png", dpi=300)
    print("Saved 2_trade_analysis.png")
    plt.close()

def plot_daily_betas_evolution(betas, trades):
    """Plots the evolution of Beta for the top 5 traded pairs."""
    if betas.empty or trades.empty: return
    
    # Identify top pairs by volume
    trades['pair_name'] = trades['pair_y'] + "-" + trades['pair_x']
    betas['pair_name'] = betas['pair_y'] + "-" + betas['pair_x']
    
    top_pairs = trades['pair_name'].value_counts().head(5).index.tolist()
    
    fig, axes = plt.subplots(len(top_pairs), 1, figsize=(12, 3*len(top_pairs)), sharex=True)
    if len(top_pairs) == 1: axes = [axes]
    
    for idx, pair in enumerate(top_pairs):
        ax = axes[idx]
        subset = betas[betas['pair_name'] == pair].sort_values('date')
        
        # Plot Beta Line
        ax.plot(subset['date'], subset['beta'], marker='o', markersize=3, label=f'Beta {pair}', color='purple')
        
        # Overlay actual trades as dots
        pair_trades = trades[trades['pair_name'] == pair]
        if not pair_trades.empty:
            # Shift x slightly to match daily grid if needed, but timestamp should work
            # Green for Win, Red for Loss
            wins = pair_trades[pair_trades['pnl'] > 0]
            losses = pair_trades[pair_trades['pnl'] <= 0]
            
            # We plot them at the Beta level recorded for that day? 
            # Or just markers on the line? Let's put markers on the Beta line.
            # We need to map trade time to the daily beta value.
            # Merge approx
            
            # Simple visualization: Vertical lines for trade activity?
            # Or just scatter on the secondary axis?
            
            # Let's plot PnL on secondary axis to see if Beta change impacted result
            ax2 = ax.twinx()
            ax2.bar(pair_trades['entry_time'], pair_trades['pnl'], width=0.05, alpha=0.5, 
                    color=np.where(pair_trades['pnl']>0, 'g', 'r'), label='Trade PnL')
            ax2.set_ylabel("Trade PnL")
            ax2.grid(False)
            
        ax.set_title(f"Hedge Ratio Evolution: {pair}")
        ax.set_ylabel("Beta")
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "3_beta_evolution.png", dpi=300)
    print("Saved 3_beta_evolution.png")
    plt.close()

def plot_positions_timeline(trades):
    """Gantt-like chart of positions."""
    if trades.empty: return
    
    # Filter to a reasonable window if too many trades, or just plot all
    # Let's plot last 100 trades if too many
    plot_trades = trades.tail(100).copy()
    plot_trades['pair_name'] = plot_trades['pair_y'] + "/" + plot_trades['pair_x']
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Unique Y position for each pair
    pairs = plot_trades['pair_name'].unique()
    y_map = {p: i for i, p in enumerate(pairs)}
    
    for _, t in plot_trades.iterrows():
        y_pos = y_map[t['pair_name']]
        color = 'g' if t['pnl'] > 0 else 'r'
        
        # Draw line from entry to exit
        ax.plot([t['entry_time'], t['exit_time']], [y_pos, y_pos], color=color, linewidth=4, solid_capstyle='round')
        
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_xlabel("Time")
    ax.set_title("Recent Trades Timeline (Green=Win, Red=Loss)")
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "4_positions_timeline.png", dpi=300)
    print("Saved 4_positions_timeline.png")
    plt.close()

def main():
    print("Generatings Plots...")
    daily, trades, betas = load_data()
    
    if daily is not None:
        plot_equity_curve(daily)
        plot_trade_analysis(trades)
        plot_daily_betas_evolution(betas, trades)
        plot_positions_timeline(trades)
        print("Done. Plots saved in pairs_trading/plots/")

if __name__ == "__main__":
    main()
