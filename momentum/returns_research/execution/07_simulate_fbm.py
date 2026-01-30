import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_fgn(n, hurst):
    """
    Generates Fractional Gaussian Noise (fGn) using the Davies-Harte method.
    This is the exact method for simulating fGn.
    """
    # Auto-covariance function of fGn
    k = np.arange(0, n + 1)
    gamma = 0.5 * (np.abs(k - 1)**(2 * hurst) - 2 * np.abs(k)**(2 * hurst) + np.abs(k + 1)**(2 * hurst))
    
    # Eigenvalues of the circulant matrix
    g = np.concatenate((gamma[:n], gamma[n-1:0:-1]))
    j = np.arange(0, 2 * n)
    # FFT of covariance
    L = np.fft.fft(g).real
    
    if np.any(L < 0):
        # Fallback (approximate) if Davies-Harte fails (rare for H < 1)
        # But for n large and H close to 0 or 1, numerical errors can happen.
        # We clip to 0.
        L = np.clip(L, 0, None)
        
    # Generate random noise in frequency domain
    z = np.random.standard_normal(2 * n)
    w = np.zeros(2 * n, dtype=complex)
    w[0] = np.sqrt(L[0] / (2 * n)) * z[0]
    w[1:n] = np.sqrt(L[1:n] / (4 * n)) * (z[1:n] + 1j * z[n+1:2*n])
    w[n] = np.sqrt(L[n] / (2 * n)) * z[n]
    w[n+1:] = np.conj(w[n-1:0:-1])
    
    # IFFT to get fGn
    fgn = np.fft.ifft(w).real * (2 * n)
    return fgn[:n]

def main():
    # Simulation Parameters
    N = 100000 # Number of points (ticks/minutes)
    SIGMA = 0.0001 # Volatility per step (approx 1bp per min)
    START_PRICE = 100.0
    
    np.random.seed(42)
    
    print(f"Simulating Fractional Brownian Motions (N={N})...")
    
    # 1. Mean Reverting (H = 0.41)
    print("Generating H=0.41 (Mean Reverting)...")
    fgn_mr = generate_fgn(N, 0.41)
    price_mr = START_PRICE + np.cumsum(fgn_mr) * START_PRICE * SIGMA
    
    # 2. Random Walk (H = 0.50)
    print("Generating H=0.50 (Random Walk)...")
    fgn_rw = np.random.standard_normal(N) # Standard Gaussian
    price_rw = START_PRICE + np.cumsum(fgn_rw) * START_PRICE * SIGMA
    
    # 3. Trending (H = 0.60)
    print("Generating H=0.60 (Trending)...")
    fgn_tr = generate_fgn(N, 0.60)
    price_tr = START_PRICE + np.cumsum(fgn_tr) * START_PRICE * SIGMA
    
    # --- Plotting ---
    plt.style.use('dark_background')
    
    # Plot 1: The Paths
    plt.figure(figsize=(16, 8))
    
    plt.plot(price_mr, label='Mean Reverting (H=0.41)', color='#00FF00', linewidth=1, alpha=0.9)
    plt.plot(price_rw, label='Random Walk (H=0.50)', color='#808080', linewidth=0.8, alpha=0.5)
    plt.plot(price_tr, label='Trending (H=0.60)', color='#FF4500', linewidth=1, alpha=0.7)
    
    plt.title(f'Simulation: Effect of Hurst Exponent on Price Path (N={N})', fontsize=16)
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Price')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.2)
    
    out_dir = "returns_research/deliverables/04_hurst_predictability"
    plt.savefig(f"{out_dir}/simulated_paths.png", dpi=300, bbox_inches='tight')
    print("Saved simulated_paths.png")
    plt.close()
    
    # Plot 2: Drawdown Analysis (Visualizing the "Pain")
    # Calculate Drawdowns
    def get_dd(p):
        peak = np.maximum.accumulate(p)
        dd = (p - peak) / peak
        return dd
        
    dd_mr = get_dd(price_mr)
    dd_rw = get_dd(price_rw)
    
    plt.figure(figsize=(16, 6))
    plt.plot(dd_mr * 100, label='Mean Reverting (H=0.41)', color='#00FF00', linewidth=0.8)
    plt.plot(dd_rw * 100, label='Random Walk (H=0.50)', color='#808080', linewidth=0.8, alpha=0.4)
    
    plt.title('Drawdown Profile: H=0.41 vs H=0.50', fontsize=16)
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.fill_between(range(N), dd_mr*100, 0, color='#00FF00', alpha=0.1)
    
    plt.savefig(f"{out_dir}/simulated_drawdowns.png", dpi=300, bbox_inches='tight')
    print("Saved simulated_drawdowns.png")
    plt.close()
    
    # Plot 3: 1-Day Returns Distribution (Fat Tails?)
    # H < 0.5 usually has thinner tails than normal? Or just anti-persistence.
    # Actually, fractional noise is Gaussian by definition (fGn). 
    # The autocorrelation structure is what differs.
    
    from statsmodels.graphics.tsaplots import plot_acf
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Autocorrelation of Returns
    # For H=0.41, lag-1 autocorrelation should be negative.
    # Theoretical rho(1) = 0.5 * (2^(2H) - 2)
    theo_rho = 0.5 * (2**(2*0.41) - 2)
    
    # MR ACF
    # Calculate ACF manually for speed/control
    # We use numpy correlate
    ret_mr = np.diff(price_mr)
    ret_mr = (ret_mr - np.mean(ret_mr)) / np.std(ret_mr)
    acf_mr = np.correlate(ret_mr, ret_mr, mode='full')[len(ret_mr)-1:] / len(ret_mr)
    acf_mr = acf_mr[:50] # First 50 lags
    
    axes[0].bar(range(50), acf_mr, color='#00FF00', alpha=0.7, label='H=0.41 Returns')
    axes[0].axhline(theo_rho, color='white', linestyle='--', label=f'Theoretical Lag-1 ({theo_rho:.4f})')
    axes[0].set_title('Autocorrelation of Returns (H=0.41)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    
    # RW ACF
    ret_rw = np.diff(price_rw)
    ret_rw = (ret_rw - np.mean(ret_rw)) / np.std(ret_rw)
    acf_rw = np.correlate(ret_rw, ret_rw, mode='full')[len(ret_rw)-1:] / len(ret_rw)
    acf_rw = acf_rw[:50]
    
    axes[1].bar(range(50), acf_rw, color='#808080', alpha=0.7, label='H=0.50 Returns')
    axes[1].set_title('Autocorrelation of Returns (H=0.50)', fontsize=14)
    axes[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/simulated_acf.png", dpi=300, bbox_inches='tight')
    print("Saved simulated_acf.png")
    plt.close()

if __name__ == "__main__":
    main()
