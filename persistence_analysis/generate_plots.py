
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
import os

# Configuration
RESULTS_FILE = Path("/mnt/ssd2/DARWINEX_Mission/results/persistence_v2_fixed_beta.pkl")
OUTPUT_DIR = Path("persistence_analysis/plots")
CRIT_VAL = -3.34

def ensure_output_dir():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if not RESULTS_FILE.exists():
        print(f"⚠️ Data file not found: {RESULTS_FILE}")
        return None
    
    with open(RESULTS_FILE, 'rb') as f:
        results = pickle.load(f)
    print(f"✅ Loaded results for windows: {list(results.keys())}")
    
    records = []
    for w_size, days_data in results.items():
        for day_entry in days_data:
            date = day_entry['date']
            for res in day_entry['data']:
                beta_next = res.get('beta_next', np.nan)
                records.append({
                    'Window': w_size,
                    'Date': date,
                    'Pair': f"{res['y']}-{res['x']}",
                    'Score_Train': res['score'],
                    'Score_Test_Fixed': res['score_next_day'],
                    'Beta_T': res['beta'],
                    'Beta_T1': beta_next,
                    'Beta_Shift': abs(beta_next - res['beta']) if not np.isnan(beta_next) else np.nan
                })
    
    df = pd.DataFrame(records)
    print(f"Total opportunities: {len(df)}")
    print(f"Records with Beta Shift data: {df['Beta_Shift'].notna().sum()}")
    return df

def plot_persistence_rate(df):
    """Generates the persistence rate bar chart."""
    df['Is_Persistent'] = df['Score_Test_Fixed'] < CRIT_VAL
    persistence_stats = df.groupby('Window')['Is_Persistent'].mean() * 100

    plt.figure(figsize=(8, 5))
    ax = persistence_stats.plot(kind='bar', color=['#4c72b0', '#55a868', '#c44e52'])
    plt.title('Persistence Rate: % of Pairs Remaining Cointegrated at T+1 (Fixed Beta)', fontsize=14)
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Window Size (Days)')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)

    for index, value in enumerate(persistence_stats):
        plt.text(index, value + 1, f"{value:.1f}%", ha='center', fontweight='bold')
    
    output_path = OUTPUT_DIR / "persistence_rate.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_score_degradation(df):
    """Generates the scatter plot comparing Train vs Test scores."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Score_Train', y='Score_Test_Fixed', hue='Window', alpha=0.3, palette='viridis', s=15)
    plt.plot([-10, 0], [-10, 0], 'r--', label='Perfect Stability (y=x)')
    plt.axhline(CRIT_VAL, color='k', linestyle=':', label='5% Crit Value')
    plt.axvline(CRIT_VAL, color='k', linestyle=':')

    plt.title('Score Degradation: Training (Optimized) vs Testing (Fixed Beta)', fontsize=14)
    plt.xlabel('Training Score (Day T)')
    plt.ylabel('Testing Score (Day T+1, Fixed Beta)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = OUTPUT_DIR / "score_degradation.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def exp_func(x, a, b, c):
    """Exponential function for curve fitting: y = a * e^(b * x) + c"""
    return a * np.exp(b * x) + c

def plot_beta_stability_analysis(df):
    """Generates Boxplot and Scatterplot with Regression for Beta Shift."""
    # Filter for valid beta shifts and remove extreme outliers for clearer visualization
    df_beta = df[df['Beta_Shift'].notna() & (df['Beta_Shift'] < 2.0)].copy()

    # --- 1. Boxplot ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_beta, x='Window', y='Beta_Shift', palette='pastel', showfliers=False)
    plt.title('Distribution of 1-Day Beta Shift by Window Size (Outliers Hidden)', fontsize=14)
    plt.ylabel('Beta Shift')
    plt.grid(axis='y', alpha=0.3)
    
    output_path_box = OUTPUT_DIR / "beta_shift_boxplot.png"
    plt.savefig(output_path_box)
    plt.close()
    print(f"Saved: {output_path_box}")

    # --- 2. Scatter with Exponential Regression (Fit to Binned Medians) ---
    plt.figure(figsize=(12, 7))

    # Scatter points (Raw Data)
    sns.scatterplot(data=df_beta, x='Score_Train', y='Beta_Shift', hue='Window', alpha=0.15, palette='rocket', s=15, legend=False)

    colors = sns.color_palette('rocket', n_colors=len(df_beta['Window'].unique()))
    windows = sorted(df_beta['Window'].unique())
    x_range = np.linspace(df_beta['Score_Train'].min(), df_beta['Score_Train'].max(), 100)

    for i, win in enumerate(windows):
        subset = df_beta[df_beta['Window'] == win].copy()
        
        # 1. Bin the data to find the trend of the median
        # Create bins across the score range
        bins = np.linspace(subset['Score_Train'].min(), subset['Score_Train'].max(), 25)
        subset['bin'] = pd.cut(subset['Score_Train'], bins)
        
        # Calculate median Beta Shift for each bin
        bin_stats = subset.groupby('bin', observed=True)['Beta_Shift'].median().reset_index()
        # Calculate bin centers
        bin_stats['bin_center'] = bin_stats['bin'].apply(lambda x: x.mid).astype(float)
        
        # Drop empty bins
        bin_stats = bin_stats.dropna()
        
        if len(bin_stats) > 3:
            x_fit = bin_stats['bin_center'].values
            y_fit = bin_stats['Beta_Shift'].values

            # Plot bin medians (optional, helps verify fit)
            plt.scatter(x_fit, y_fit, color=colors[i], s=40, marker='x', alpha=0.8) #, label=f'Medians (W={win})')

            try:
                # Fit exponential to the MEDIANS
                # y = a * exp(b * x) + c
                # Expect b > 0 (growth as x goes from -10 to -3)
                p0 = [0.01, 0.5, 0.0] 
                popt, _ = curve_fit(exp_func, x_fit, y_fit, p0=p0, maxfev=5000)
                
                label = f'Trend W={win}'
                plt.plot(x_range, exp_func(x_range, *popt), color=colors[i], linewidth=3, label=label)
            except Exception as e:
                print(f"Fit failed for W={win}: {e}. using Poly.")
                z = np.polyfit(x_fit, y_fit, 2)
                p = np.poly1d(z)
                plt.plot(x_range, p(x_range), color=colors[i], linestyle='--', linewidth=3, label=f'Poly W={win}')

    plt.title('Correlation: Cointegration Strength vs. Beta Stability (Median Trend)', fontsize=14)
    plt.xlabel('Cointegration Strength (Score at T)')
    plt.ylabel('Beta Shift (Next Day)')
    plt.ylim(0, 1.0) # Zoom in a bit to see the curve better
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[i], lw=3) for i in range(len(windows))]
    plt.legend(custom_lines, [f'Window {w}' for w in windows], title="Window Size")
    
    plt.grid(True, alpha=0.3)

    output_path_scatter = OUTPUT_DIR / "beta_stability_regression.png"
    plt.savefig(output_path_scatter)
    plt.close()
    print(f"Saved: {output_path_scatter}")

def main():
    ensure_output_dir()
    df = load_data()
    if df is not None:
        plot_persistence_rate(df)
        plot_score_degradation(df)
        plot_beta_stability_analysis(df)
        print("All plots generated successfully.")

if __name__ == "__main__":
    main()
