from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig, simulate_split_trade
from backtesting.v4_2_causal import _variant_frame
from backtesting.v4_4_robustness import RobustnessConfig, rolling_windows, robustness_summary


@dataclass(frozen=True)
class EntryRetestConfig:
    retest_window_bars: int = 24
    fvg_mid_fraction: float = 0.50
    ote_levels: tuple[float, ...] = (0.62, 0.705, 0.79)
    min_gap_points: float = 0.0

    def __post_init__(self) -> None:
        if self.retest_window_bars <= 0:
            raise ValueError("retest_window_bars must be positive")
        if not 0.0 <= self.fvg_mid_fraction <= 1.0:
            raise ValueError("fvg_mid_fraction must be between 0 and 1")
        if any(not 0.0 < x < 1.0 for x in self.ote_levels):
            raise ValueError("OTE levels must be between 0 and 1")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _valid_entry(direction: str, entry: float, stop: float) -> bool:
    return entry > stop if direction == "BULLISH" else entry < stop


def _fvg_zone_at_confirmation(m5: pd.DataFrame, confirmation_close: pd.Timestamp, direction: str) -> dict | None:
    """Return the three-candle M5 FVG that is known at confirmation_close.

    MT5 timestamps are bar-open times. A five-minute FVG is known only when its
    third candle closes, so confirmation_close must equal third_open + 5 min.
    """
    third_open = pd.Timestamp(confirmation_close) - pd.to_timedelta(5, unit="min")
    idx = m5.index[m5["time"].eq(third_open)]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i < 2:
        return None
    first = m5.iloc[i - 2]
    third = m5.iloc[i]
    if direction == "BULLISH":
        low = float(first["high"])
        high = float(third["low"])
        impulse_extreme = float(third["high"])
        valid = high > low
    else:
        low = float(third["high"])
        high = float(first["low"])
        impulse_extreme = float(third["low"])
        valid = high > low
    if not valid:
        return None
    return {"fvg_low": low, "fvg_high": high, "gap_size": high - low, "impulse_extreme": impulse_extreme}


def _limit_fill(
    m5: pd.DataFrame,
    confirmation_close: pd.Timestamp,
    direction: str,
    entry: float,
    stop: float,
    max_bars: int,
) -> tuple[pd.Timestamp | pd.NaT, int, str]:
    """Find the first post-confirmation M5 limit fill with conservative invalidation."""
    future = m5.loc[m5["time"] >= pd.Timestamp(confirmation_close)].head(max_bars)
    for n, (_, bar) in enumerate(future.iterrows(), start=1):
        high, low = float(bar["high"]), float(bar["low"])
        stop_touched = low <= stop if direction == "BULLISH" else high >= stop
        fill_touched = low <= entry <= high
        if stop_touched:
            # If stop and entry are both touched in one bar we cannot know order;
            # invalidate conservatively instead of granting a fill.
            return pd.NaT, n, "STOP_BEFORE_FILL"
        if fill_touched:
            return pd.Timestamp(bar["time"]), n, "FILLED"
    return pd.NaT, int(len(future)), "NO_RETEST"


def _simulate_after_fill(
    m5: pd.DataFrame,
    fill_bar_open: pd.Timestamp,
    direction: str,
    entry: float,
    stop: float,
    split_cfg: SplitTargetConfig,
) -> dict | None:
    if pd.isna(fill_bar_open) or not _valid_entry(direction, entry, stop):
        return None
    # Entry may occur anywhere within the fill candle. To avoid optimistic
    # same-bar ordering, exits start from the next completed M5 candle.
    next_open = pd.Timestamp(fill_bar_open) + pd.to_timedelta(5, unit="min")
    future = m5.loc[m5["time"] >= next_open].copy()
    if future.empty:
        return None
    return simulate_split_trade(future, direction, entry, stop, split_cfg)


def _ote_entry(direction: str, stop: float, impulse_extreme: float, retracement: float) -> float:
    distance = abs(impulse_extreme - stop)
    if direction == "BULLISH":
        return impulse_extreme - retracement * distance
    return impulse_extreme + retracement * distance


def build_entry_retest_trades(
    trades: pd.DataFrame,
    m5: pd.DataFrame,
    config: EntryRetestConfig | None = None,
    split_config: SplitTargetConfig | None = None,
) -> pd.DataFrame:
    cfg = config or EntryRetestConfig()
    split_cfg = split_config or SplitTargetConfig(max_holding_bars=72)
    bars = _normalize(m5)
    base = _variant_frame(trades, "FULL_M5_CAUSAL").copy()
    if base.empty:
        return pd.DataFrame()
    base["signal_close_time_utc"] = pd.to_datetime(base["signal_close_time_utc"], utc=True, errors="coerce")
    base["m5_fvg_time"] = pd.to_datetime(base["m5_fvg_time"], utc=True, errors="coerce")

    rows: list[dict] = []
    for _, r in base.iterrows():
        conf_time = r.get("m5_fvg_time")
        if pd.isna(conf_time):
            continue
        direction = str(r["direction"])
        stop = float(r["stop"])
        zone = _fvg_zone_at_confirmation(bars, pd.Timestamp(conf_time), direction)
        if zone is None or float(zone["gap_size"]) < cfg.min_gap_points:
            continue

        fvg_mid = float(zone["fvg_low"] + cfg.fvg_mid_fraction * zone["gap_size"])
        entries = {"FVG_MID": fvg_mid}
        for level in cfg.ote_levels:
            entries[f"OTE_{str(level).replace('.', '_')}"] = _ote_entry(direction, stop, float(zone["impulse_extreme"]), float(level))

        for entry_model, entry in entries.items():
            if not _valid_entry(direction, float(entry), stop):
                continue
            fill_time, wait_bars, fill_status = _limit_fill(
                bars, pd.Timestamp(conf_time), direction, float(entry), stop, cfg.retest_window_bars
            )
            result = _simulate_after_fill(bars, fill_time, direction, float(entry), stop, split_cfg)
            row = {
                "entry_model": entry_model,
                "signal_close_time_utc": r["signal_close_time_utc"],
                "confirmation_time_utc": pd.Timestamp(conf_time),
                "fill_bar_open_utc": fill_time,
                "direction": direction,
                "ny_hour": r.get("ny_hour"),
                "year": r.get("year"),
                "stop": stop,
                "entry": float(entry),
                "fvg_low": float(zone["fvg_low"]),
                "fvg_high": float(zone["fvg_high"]),
                "fvg_gap_size": float(zone["gap_size"]),
                "impulse_extreme": float(zone["impulse_extreme"]),
                "wait_bars": int(wait_bars),
                "fill_status": fill_status,
            }
            if result is not None:
                row.update(result)
            rows.append(row)
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame) -> dict:
    filled = frame.loc[frame["fill_status"].eq("FILLED") & frame.get("total_r", pd.Series(index=frame.index, dtype=float)).notna()].copy()
    if filled.empty:
        return {"setups": int(len(frame)), "filled": 0, "fill_rate": 0.0, "total_r": 0.0, "expectancy_r": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0}
    filled = filled.sort_values("signal_close_time_utc")
    r = filled["total_r"].astype(float)
    wins = float(r[r > 0].sum()); losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    eq = r.cumsum().reset_index(drop=True)
    peak = pd.concat([pd.Series([0.0]), eq]).cummax().iloc[1:].reset_index(drop=True)
    dd = eq - peak
    return {
        "setups": int(len(frame)), "filled": int(len(filled)), "fill_rate": float(len(filled) / len(frame)),
        "total_r": float(r.sum()), "expectancy_r": float(r.mean()), "win_rate": float((r > 0).mean()),
        "profit_factor": float(pf), "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
    }


def run_v4_6_entry_retest(
    trades: pd.DataFrame,
    m5: pd.DataFrame,
    output_dir: str | Path = "data/research/v4_6_entry_retest",
    config: EntryRetestConfig | None = None,
    split_config: SplitTargetConfig | None = None,
):
    results = build_entry_retest_trades(trades, m5, config, split_config)
    summary_rows = []
    rolling_rows = []
    rcfg = RobustnessConfig(window_days=90, step_days=30, min_trades_per_window=15)

    if not results.empty:
        for model, grp in results.groupby("entry_model", sort=False):
            m = _metrics(grp)
            filled = grp.loc[grp["fill_status"].eq("FILLED") & grp.get("total_r", pd.Series(index=grp.index, dtype=float)).notna()].copy()
            if not filled.empty:
                w = rolling_windows(filled, rcfg)
                rs = robustness_summary(filled, w, model, rcfg)
                m.update({
                    "windows": rs["windows"],
                    "positive_window_share": rs["positive_window_share"],
                    "median_window_expectancy": rs["median_window_expectancy"],
                    "stable": rs["stable"],
                })
                if not w.empty:
                    w = w.copy(); w["entry_model"] = model; rolling_rows.append(w)
            else:
                m.update({"windows": 0, "positive_window_share": np.nan, "median_window_expectancy": np.nan, "stable": False})
            summary_rows.append({"entry_model": model, **m})

    summary = pd.DataFrame(summary_rows)
    rolling = pd.concat(rolling_rows, ignore_index=True) if rolling_rows else pd.DataFrame()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "v4_6_entry_retest_trades.csv", index=False)
    summary.to_csv(out / "v4_6_entry_retest_summary.csv", index=False)
    rolling.to_csv(out / "v4_6_entry_retest_rolling.csv", index=False)
    return summary, rolling, results
