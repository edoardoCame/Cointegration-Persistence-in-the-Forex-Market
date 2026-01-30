import os
import glob
import cudf
import numpy as np

# Configuration
DATA_DIR = "/mnt/ssd2/DARWINEX_Mission/data"

def get_all_symbols():
    """Scans data dir for all available symbols."""
    files = glob.glob(os.path.join(DATA_DIR, "DAT_ASCII_*_M1_*.csv"))
    symbols = set()
    for f in files:
        # Extract symbol: DAT_ASCII_EURUSD_M1_2025.csv -> EURUSD
        parts = os.path.basename(f).split('_')
        if len(parts) >= 3:
            symbols.add(parts[2])
    return sorted(list(symbols))

def load_data(symbol, timeframe='15min'):
    """Loads and concatenates data for a symbol, resamples to timeframe."""
    pattern = os.path.join(DATA_DIR, f"DAT_ASCII_{symbol}_M1_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"Warning: No files found for {symbol}")
        return None
        
    dfs = []
    for f in files:
        try:
            # Read CSV with cudf
            df = cudf.read_csv(
                f, 
                sep=';', 
                names=['Date Time', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            df['Date Time'] = cudf.to_datetime(df['Date Time'], format='%Y%m%d %H%M%S')
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return None
        
    full_df = cudf.concat(dfs)
    full_df = full_df.sort_values('Date Time')
    full_df = full_df.set_index('Date Time')
    
    # Resample
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    if len(full_df) == 0:
        return None

    resampled = full_df.resample(timeframe).agg(agg_dict)
    resampled = resampled.dropna()
    
    return resampled
