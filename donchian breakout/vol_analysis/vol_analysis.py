"""
GPU-powered EURUSD hourly momentum/vol profile by hour-of-day.
Reads minute data, aggregates to hourly closes, computes hourly returns, then for each hour (0-23):
- cumulative return (absolute value) for that hour across the sample
- std dev of returns for that hour
- metric = abs(cum return) * std, normalized by its mean, shown as a bar plot
Plots and CSV are saved into the plots/ folder next to this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cudf
import cupy as cp  # noqa: F401  # reserved for potential GPU math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_hourly_returns(csv_path: Path) -> cudf.Series:
    """Load minute bars, aggregate to hourly closes, and compute hourly pct returns."""
    col_names = ["datetime", "open", "high", "low", "close", "volume"]
    df = cudf.read_csv(
        csv_path,
        names=col_names,
        sep=";",
        header=None,
        dtype={"open": "float32", "high": "float32", "low": "float32", "close": "float32", "volume": "int64"},
    )
    df["timestamp"] = cudf.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")
    df = df.sort_values("timestamp")

    # cudf 23.12 lacks dt.floor("h"); build an hourly key via integer nanoseconds
    ns_per_hour = 3_600 * 1_000_000_000
    ts_ns = df["timestamp"].astype("int64")
    hourly_idx = cudf.to_datetime((ts_ns // ns_per_hour) * ns_per_hour)
    hourly_close = df.groupby(hourly_idx)["close"].last()
    hourly_returns = hourly_close.pct_change()
    return hourly_returns


def build_hour_profile(hourly_returns: cudf.Series) -> cudf.DataFrame:
    """For each hour-of-day (0-23), compute cumulative return, std, and a normalized score."""
    hr = hourly_returns.dropna()
    df = cudf.DataFrame({"timestamp": hr.index, "ret": hr.values})
    df["hour"] = df["timestamp"].dt.hour

    rows = []
    for h in range(24):
        r = df.loc[df["hour"] == h, "ret"]
        if len(r) == 0:
            rows.append({"hour": h, "cum_return_pct": cp.nan, "vol_pct": cp.nan, "metric": cp.nan})
            continue
        cum_ret = cp.abs((1 + r).prod() - 1)  # absolute cumulative return
        vol = r.std()
        metric = cum_ret * vol
        rows.append(
            {
                "hour": h,
                "cum_return_pct": float(cum_ret * 100),
                "vol_pct": float(vol * 100),
                "metric": float(metric),
            }
        )

    out = cudf.DataFrame(rows)
    out["metric_norm"] = out["metric"] / out["metric"].mean()
    return out


def _format_dates(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.grid(True, alpha=0.3)


def plot_series(series, title: str, ylabel: str, out_path: Path):
    """Plot a single time series and save it."""
    ps = series.to_pandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ps.index, ps.values, lw=1.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hour_profile(hour_df: cudf.DataFrame, out_path: Path):
    """Bar plot of the normalized hour-of-day metric."""
    pdf = hour_df.to_pandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(pdf["hour"], pdf["metric_norm"], color="#4a90e2", alpha=0.85)

    # Evidenzia le sessioni FX (fuso orario: EST)
    sessions = [
        ("Sydney", [(17, 24), (0, 2)], "#b3cde0"),
        ("Tokyo", [(19, 24), (0, 4)], "#ccebc5"),
        ("London", [(3, 12)], "#decbe4"),
        ("New York", [(8, 17)], "#fed9a6"),
    ]

    for name, spans, color in sessions:
        for start, end in spans:
            start_edge = start - 0.5
            end_edge = (end - 0.5) if end < 24 else 24.0
            ax.axvspan(start_edge, end_edge, color=color, alpha=0.25, lw=0)

    legend_patches = [Patch(color=c, alpha=0.5, label=n) for n, _, c in sessions]
    ax.axhline(1.0, color="black", lw=1, linestyle="--", label="Media")
    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("Ora del giorno (EST)")
    ax.set_ylabel("Metric (|cum return| * std) / media")
    ax.set_title("Profilo orario FX (EST): momentum-vol normalizzato")
    ax.legend(handles=legend_patches + [ax.lines[-1]])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dual(series_a, series_b, labels: tuple[str, str], title: str, out_path: Path):
    """Plot two aligned series for quick comparison."""
    pa = series_a.to_pandas()
    pb = series_b.to_pandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pa.index, pa.values, label=labels[0], lw=1.1)
    ax.plot(pb.index, pb.values, label=labels[1], lw=1.1, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Percent")
    ax.legend()
    _format_dates(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(input_csv: Path, output_dir: Path, roll_hours: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_returns = load_hourly_returns(input_csv)
    hour_profile = build_hour_profile(hourly_returns)

    plot_hour_profile(
        hour_profile,
        out_path=output_dir / "hour_profile_metric_norm.png",
    )

    hour_profile.to_pandas().to_csv(output_dir / "hour_profile.csv", index=False)


if __name__ == "__main__":
    default_input = Path(__file__).resolve().parents[2] / "data" / "DAT_ASCII_EURUSD_M1_2025.csv"
    default_output = Path(__file__).resolve().parent / "plots"

    parser = argparse.ArgumentParser(description="GPU EURUSD hourly momentum and vol analysis")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to minute-level CSV (EURUSD)")
    parser.add_argument("--output", type=Path, default=default_output, help="Directory for output plots")
    parser.add_argument("--roll-hours", type=int, default=24, help="Rolling window (hours) for volatility")
    args = parser.parse_args()

    run(args.input, args.output, args.roll_hours)
