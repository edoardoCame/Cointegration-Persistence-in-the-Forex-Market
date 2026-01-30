import sys
import os

# Add parent directory to path to find 'library'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import cupy as cp
import cudf
import plotly.express as px
import plotly.graph_objects as go
from system.data_feed import load_data, get_all_symbols
import warnings
import networkx as nx

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../deliverables/01_market_maps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis Params
MAX_LAG = 24 # 24 * 15m = 6 hours
TIMEFRAME = '15min'

def run_cross_asset_predictive_power(symbols, lookback=60, horizon=15):
    """
    Calculates Cross-Asset Predictive Power.
    Predictor: Asset A's Rolling Return (Lookback).
    Target: Asset B's Future Rolling Return (Horizon).
    Returns Matrix: Row(A) predicting Col(B).
    """
    print(f"Building Cross-Asset Predictive Matrix (L:{lookback}m -> H:{horizon}m)...")
    
    merged_predictors = cudf.DataFrame()
    merged_targets = cudf.DataFrame()
    
    lb_periods = int(lookback / 15)
    h_periods = int(horizon / 15)
    
    for sym in symbols:
        df = load_data(sym, TIMEFRAME)
        if df is None: continue
        
        # Log Returns
        log_ret_series = np.log(df['Close'] / df['Close'].shift(1))
        
        # Predictor: Rolling Sum (Lookback)
        pred = log_ret_series.rolling(lb_periods).sum()
        
        # Target: Future Rolling Sum (Horizon)
        # Shifted back so Target[T] aligns with Predictor[T]
        targ = log_ret_series.rolling(h_periods).sum().shift(-h_periods)
        
        # Rename and Join
        p_df = pred.to_frame(name=sym)
        t_df = targ.to_frame(name=sym)
        
        if len(merged_predictors) == 0:
            merged_predictors = p_df
            merged_targets = t_df
        else:
            merged_predictors = merged_predictors.join(p_df, how='outer')
            merged_targets = merged_targets.join(t_df, how='outer')
            
    # Align and Drop NaNs
    common_index = merged_predictors.index.intersection(merged_targets.index)
    merged_predictors = merged_predictors.loc[common_index]
    merged_targets = merged_targets.loc[common_index]
    
    merged_predictors = merged_predictors.dropna()
    merged_targets = merged_targets.loc[merged_predictors.index]
    
    merged_targets = merged_targets.dropna()
    merged_predictors = merged_predictors.loc[merged_targets.index]
    
    if len(merged_predictors) < 100:
        return pd.DataFrame()

    # Move to CuPy
    X_mat = cp.asarray(merged_predictors.values)
    Y_mat = cp.asarray(merged_targets.values)
    cols = merged_predictors.columns.to_list()
    
    # Normalize
    X_mean = X_mat.mean(axis=0)
    X_std = X_mat.std(axis=0)
    X_norm = (X_mat - X_mean) / (X_std + 1e-9)
    
    Y_mean = Y_mat.mean(axis=0)
    Y_std = Y_mat.std(axis=0)
    Y_norm = (Y_mat - Y_mean) / (Y_std + 1e-9)
    
    # Correlation Matrix = (X_norm.T @ Y_norm) / N
    N = X_norm.shape[0]
    corr_matrix = cp.dot(X_norm.T, Y_norm) / N
    
    return pd.DataFrame(cp.asnumpy(corr_matrix), index=cols, columns=cols)

def plot_lead_lag_network(matrix_df):
    """
    Visualizes the Cross-Asset Predictive Structure including all assets.
    Top 50 strongest edges are shown with variable opacity for readability.
    """
    print("Generating Full Lead-Lag Network Graph (All Assets, Top 50 Edges)...")
    
    # Flatten and sort all potential edges
    edges = []
    for predictor in matrix_df.index:
        for target in matrix_df.columns:
            if predictor == target: continue
            weight = matrix_df.loc[predictor, target]
            edges.append((predictor, target, weight))
            
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    top_edges = edges[:50] # Increased to 50 for a fuller picture
    
    G = nx.DiGraph()
    
    # ALWAYS Add All Nodes (even if isolated)
    for sym in matrix_df.index:
        G.add_node(sym)
        
    # Add Top Edges
    for pred, targ, w in top_edges:
        G.add_edge(pred, targ, weight=w)
    
    # Layout - Spring layout groups related nodes
    pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), iterations=100, seed=42)
    
    annotations = []
    # Max weight for normalization
    max_w = max([abs(e[2]) for e in top_edges]) if top_edges else 1
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        # Color based on sign
        color = '#00CC96' if weight > 0 else '#EF553B' 
        
        # Scale opacity and width by strength
        strength = abs(weight) / max_w
        opacity = 0.2 + (strength * 0.7) # Range 0.2 to 0.9
        width = 1 + (strength * 4)       # Range 1 to 5
        
        annotations.append(dict(
            ax=x0, ay=y0, axref='x', ayref='y',
            x=x1, y=y1, xref='x', yref='y',
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=width,
            arrowcolor=color, opacity=opacity
        ))

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        out_d = out_degrees.get(node, 0)
        in_d = in_degrees.get(node, 0)
        
        node_color.append(out_d) # Leadership
        node_size.append(15 + (in_d * 8)) # Predictability

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"<b>{n}</b>" for n in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(text='Leadership (Out-Degree)', side='right'),
                xanchor='left'
            ),
            line_width=1.5,
            line_color='rgba(255,255,255,0.5)'))

    fig = go.Figure(data=[node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='<b>Complete Cross-Asset Predictive Network</b><br>Top 50 Links | Color: Leadership | Size: Predictability',
                            font=dict(size=20)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=40,l=40,r=40,t=80),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        template='plotly_dark',
                        width=1400, height=1400
                    ))
                    
    fig.write_image(os.path.join(OUTPUT_DIR, "lead_lag_network.png"))
    print(f"Saved Full Network Graph to {OUTPUT_DIR}")

def run_lagged_autocorr_gpu(symbols):
    """
    Calculates autocorrelation for each symbol for lags 1..MAX_LAG using GPU.
    Returns a DataFrame: Index=Symbol, Columns=Lags.
    """
    results = []
    
    for sym in symbols:
        df = load_data(sym, TIMEFRAME)
        if df is None: continue
        
        # Log Returns
        log_ret_series = np.log(df['Close'] / df['Close'].shift(1))
        log_ret = log_ret_series.dropna().to_cupy()
        
        row = {'Symbol': sym}
        n = len(log_ret)
        
        for lag in range(1, MAX_LAG + 1):
            x = log_ret[lag:]
            y = log_ret[:-lag]
            
            if len(x) < 100:
                c = 0.0
            else:
                c = cp.corrcoef(x, y)[0, 1]
            
            row[lag] = float(c)
            
        results.append(row)
        
    return pd.DataFrame(results).set_index('Symbol')

def main():
    symbols = get_all_symbols()
    print(f"Starting GPU Granger/Lag Analysis on {len(symbols)} pairs...")
    
    # 1. Autocorrelation Analysis (Univariate Memory)
    df_autocorr = run_lagged_autocorr_gpu(symbols)
    
    fig_heat = px.imshow(
        df_autocorr,
        title=f"<b>Momentum Memory (Autocorrelation)</b><br>Lag 1 to {MAX_LAG} ({TIMEFRAME} bars)",
        labels=dict(x="Lag", y="Symbol", color="Correlation"),
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        aspect="auto"
    )
    fig_heat.update_layout(template='plotly_dark')
    fig_heat.write_image(os.path.join(OUTPUT_DIR, "autocorr_heatmap.png"))
    
    # 2. Cross-Asset Predictive Power
    cross_matrix = run_cross_asset_predictive_power(symbols, lookback=60, horizon=15)
    
    fig_cross = px.imshow(
        cross_matrix,
        title="<b>Cross-Asset Predictive Matrix (L:60m -> H:15m)</b><br>Row (60m) predicting Column (15m Future)",
        labels=dict(x="Target Asset (15m Future)", y="Predictor Asset (60m Past)", color="Correlation"),
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        height=900, width=900
    )
    fig_cross.update_layout(template='plotly_dark')
    fig_cross.write_image(os.path.join(OUTPUT_DIR, "cross_asset_predictivity_60m_15m.png"))
    
    # 3. Network Graph
    plot_lead_lag_network(cross_matrix)
    
    # 4. Decile Analysis (Per Symbol)
    print("Generating Decile Analysis Plots...")
    DECILE_DIR = os.path.join(OUTPUT_DIR, "deciles")
    os.makedirs(DECILE_DIR, exist_ok=True)
    
    for sym in symbols:
        df = load_data(sym, TIMEFRAME)
        if df is None: continue
        plot_decile_analysis(df, sym, DECILE_DIR)
    
    print(f"Saved all analysis to {OUTPUT_DIR}")

def plot_decile_analysis(df, symbol, output_dir):
    """
    Plots the relationship between 60m Momentum Deciles and 15m Future Returns.
    """
    try:
        # Calculate Returns
        log_ret_series = np.log(df['Close'] / df['Close'].shift(1))
        
        # 60m Rolling Momentum (4 periods of 15m)
        mom_60m = log_ret_series.rolling(4).sum()
        
        # 15m Future Return (Next period)
        fwd_15m = log_ret_series.shift(-1)
        
        # Combine and Drop NaNs
        # We need pandas for qcut usually, or cudf.qcut
        # Let's use pandas for the plotting logic ease
        data = pd.DataFrame({
            'mom': mom_60m.to_pandas(), 
            'ret': fwd_15m.to_pandas()
        }).dropna()
        
        if len(data) < 100: return

        # Create Deciles
        data['decile'] = pd.qcut(data['mom'], 10, labels=False, duplicates='drop')
        
        # Mean Return per Decile (in bps)
        means = data.groupby('decile')['ret'].mean() * 10000
        
        # Plot
        fig = px.bar(
            means,
            title=f"<b>{symbol}: Momentum Decile Analysis (60m -> 15m)</b><br>Mean Future Return (bps) per Momentum Decile",
            labels={'decile': 'Momentum Decile (0=Low, 9=High)', 'value': 'Mean Return (bps)'},
            color=means.values,
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        fig.update_layout(template='plotly_dark', showlegend=False)
        fig.add_hline(y=0, line_color="white", opacity=0.5)
        
        fig.write_image(os.path.join(output_dir, f"decile_{symbol}.png"))
        
    except Exception as e:
        print(f"Error plotting decile for {symbol}: {e}")

if __name__ == "__main__":
    main()