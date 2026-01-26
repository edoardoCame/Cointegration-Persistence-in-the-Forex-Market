import cudf
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_first_paths(pq_path, n_paths=200):
    if not os.path.exists(pq_path):
        print(f"Error: {pq_path} not found.")
        return

    print(f"Loading first {n_paths} paths from {pq_path}...")
    import pyarrow.parquet as pq
    table_meta = pq.read_metadata(pq_path)
    all_cols = table_meta.schema.names
    path_cols = [c for c in all_cols if c.startswith('path_')][:n_paths]
    
    # We don't even need the timestamp if we want to remove gaps
    df = cudf.read_parquet(pq_path, columns=path_cols)
    
    # Convert to pandas for plotting
    pdf = df.to_pandas()
    x_axis = np.arange(len(pdf))
    
    plt.figure(figsize=(15, 8))
    
    # Use a colormap for variety
    colors = plt.cm.viridis(np.linspace(0, 1, n_paths))
    
    for i, col in enumerate(path_cols):
        plt.plot(x_axis, pdf[col], alpha=0.15, color=colors[i], linewidth=0.6)
    
    plt.title(f"First {n_paths} Bootstrapped Paths (No Weekend Gaps - Weekly Shuffle)")
    plt.xlabel("Minute Index (Continuous)")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Force a tight layout to make it look cleaner
    plt.tight_layout()
    
    output_path = "BB backtest/montecarlo/first_200_paths.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_first_paths("BB backtest/montecarlo/bootstrap_paths.parquet")