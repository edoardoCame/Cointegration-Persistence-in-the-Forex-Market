# 💾 Data Directory

This folder contains the raw market data used for backtesting and analysis.

## Format Specification
The system expects **1-Minute (M1)** OHLCV data in ASCII CSV format.

*   **Filename Pattern**: `DAT_ASCII_{SYMBOL}_M1_{YEAR}.csv`
    *   Example: `DAT_ASCII_EURUSD_M1_2025.csv`
*   **Columns**: `Date Time;Open;High;Low;Close;Volume`
    *   Separator: `;`
    *   Date Format: `YYYYMMDD HHMMSS`

## Handling
*   **Loading**: The scripts use `cudf.read_csv` for ultra-fast GPU loading.
*   **Preprocessing**: Data is aligned by timestamp. Missing values are forward-filled (`ffill`) to handle small gaps in liquidity.
