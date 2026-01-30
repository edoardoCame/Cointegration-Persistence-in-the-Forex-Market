import numpy as np
import matplotlib.pyplot as plt

def simulate_ou_process(theta, mu, sigma, n_steps, dt=1.0, start_price=None):
    """
    Simulates an Ornstein-Uhlenbeck process.
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    """
    if start_price is None:
        x = np.zeros(n_steps)
        x[0] = mu
    else:
        x = np.zeros(n_steps)
        x[0] = start_price
        
    # Generate random noise
    dw = np.random.normal(0, np.sqrt(dt), n_steps)
    
    for t in range(1, n_steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * dw[t]
        x[t] = x[t-1] + dx
        
    return x

def main():
    # Parameters
    N_STEPS = 10000 # ~1 week of minutes (1440 * 7)
    MU = 1.0000     # Equilibrium Price
    SIGMA = 0.00015 # Volatility per minute (~1.5 pips)
    DT = 1.0        # 1 minute steps
    
    # Representative Thetas from the Deciles (approx mid-points)
    # D7: 3.28e-3 to 4.09e-3 -> ~0.0037
    # D8: 4.09e-3 to 5.56e-3 -> ~0.0048
    # D9: 5.56e-3 to 21.1e-3 -> ~0.0120 (taking a value in the lower-mid D9)
    
    thetas = {
        'Random Walk (Theta=0)': 0.0,
        'D7 (Theta=0.0037, HL~3h)': 0.0037,
        'D8 (Theta=0.0048, HL~2.5h)': 0.0048,
        'D9 (Theta=0.0120, HL~1h)': 0.0120
    }
    
    np.random.seed(42) # For reproducibility
    
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 8))
    
    colors = ['#808080', '#FFA500', '#FFD700', '#00FF00'] # Grey, Orange, Gold, Green
    
    for (label, theta), color in zip(thetas.items(), colors):
        print(f"Simulating {label}...")
        path = simulate_ou_process(theta, MU, SIGMA, N_STEPS, DT, start_price=MU)
        
        # Plot
        # Add slight transparency to RW to make others pop
        alpha = 0.5 if theta == 0 else 0.9
        linewidth = 1 if theta == 0 else 1.2
        
        plt.plot(path, label=label, color=color, alpha=alpha, linewidth=linewidth)
        
    plt.axhline(MU, color='white', linestyle='--', alpha=0.3, label='Equilibrium (Mu)')
    
    plt.title(f'Simulation of High-Theta Regimes (Deciles 7-9) vs Random Walk\nN={N_STEPS} minutes (~1 Trading Week)', fontsize=16)
    plt.ylabel('Price')
    plt.xlabel('Time (Minutes)')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.2)
    
    out_dir = "returns_research/deliverables/04_hurst_predictability"
    plt.savefig(f"{out_dir}/simulated_theta_paths.png", dpi=300, bbox_inches='tight')
    print("Saved simulated_theta_paths.png")
    plt.close()

if __name__ == "__main__":
    main()
