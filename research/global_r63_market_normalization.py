"""
Global r63 Market Normalization Wrapper Module.

Status: REFERENCE_ONLY / not approved for live trading.

This module dynamically imports /tmp/global_r63_reference.py, reuses a single
universe/price/FX/quality pipeline, and executes four scenarios:
  1. base_raw: S&P500 + JPX400 using raw r63 JPY total-price proxy returns
  2. all5_raw: S&P500 + JPX400 + Korea + Europe + Hong Kong + Canada + Taiwan using raw r63 JPY returns
  3. all5_market_percentile: Same all5 universe using per-date within-sleeve percentile ranks
  4. no_canada_tsm_usa_sleeve_market_percentile: Exclude Canada & scanner Taiwan, add explicit TSM ADR (market='taiwan_us', ranking sleeve='usa') using within-sleeve percentile ranks

Sleeve Mapping:
  - All 15 EU_QUOTAS markets (UK, Germany, France, Switzerland, Netherlands,
    Sweden, Denmark, Norway, Finland, Italy, Spain, Belgium, Austria, Poland,
    Portugal) map to "europe".
  - America / USA -> "usa"
  - Japan -> "japan"
  - Korea -> "korea"
  - Hong Kong -> "hong_kong"
  - Canada -> "canada"
  - Taiwan -> "taiwan"
  - Bonds -> "bond"
  - TSM candidate ranking sleeve override -> "usa"

Sorting & Tie-Breaking:
  - Deterministic sort key: (-score, -raw, ticker)
    where score is percentile rank for percentile scenarios, or raw r63 for raw scenarios.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_COMPRESSED_SHA256 = "01ef00e25838198f0772a3929518edcd9c250799d8109924ef49229a84ce3f3b"
EXPECTED_DECODED_SHA256 = "9d9a461881f0cf2874e5239a8c73334ddde004b39eea2f9bab4b2bf9a1598c45"


def _load_reference():
    """
    Dynamically loads /tmp/global_r63_reference.py.
    Numerically orders payload parts and verifies SHA256 hashes of compressed and decoded bytes.
    Hard-fails if pre-existing /tmp/global_r63_reference.py has the wrong decoded hash.
    """
    ref_path = Path("/tmp/global_r63_reference.py")
    if ref_path.exists():
        existing_sha = hashlib.sha256(ref_path.read_bytes()).hexdigest()
        if existing_sha != EXPECTED_DECODED_SHA256:
            raise ValueError(
                f"Pre-existing reference script at {ref_path} decoded hash mismatch: "
                f"expected {EXPECTED_DECODED_SHA256}, got {existing_sha}"
            )
    else:
        repo_dir = Path(__file__).resolve().parent
        part_files = sorted(
            repo_dir.glob("global_r63_reference.py.gz.b64.part*"),
            key=lambda p: int(re.search(r'part(\d+)$', p.name).group(1)) if re.search(r'part(\d+)$', p.name) else p.name
        )
        b64_file = repo_dir / "global_r63_reference.py.gz.b64"

        compressed_data = None
        if part_files:
            b64_data = b"".join(p.read_bytes() for p in part_files)
            compressed_data = base64.b64decode(b64_data)
        elif b64_file.exists():
            b64_data = b64_file.read_bytes()
            compressed_data = base64.b64decode(b64_data)

        if compressed_data is None:
            raise FileNotFoundError(
                f"Cannot find reference script at {ref_path} and no b64 payload found in {repo_dir}"
            )

        c_sha = hashlib.sha256(compressed_data).hexdigest()
        if c_sha != EXPECTED_COMPRESSED_SHA256:
            raise ValueError(
                f"Compressed payload SHA256 mismatch: expected {EXPECTED_COMPRESSED_SHA256}, got {c_sha}"
            )

        decoded_bytes = gzip.decompress(compressed_data)
        d_sha = hashlib.sha256(decoded_bytes).hexdigest()
        if d_sha != EXPECTED_DECODED_SHA256:
            raise ValueError(
                f"Decoded script SHA256 mismatch: expected {EXPECTED_DECODED_SHA256}, got {d_sha}"
            )

        ref_path.write_bytes(decoded_bytes)

    spec = importlib.util.spec_from_file_location("global_r63_reference", ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {ref_path}")
    ref_mod = importlib.util.module_from_spec(spec)
    sys.modules["global_r63_reference"] = ref_mod
    spec.loader.exec_module(ref_mod)
    return ref_mod


def get_broad_sleeve(market: str, eu_quotas: dict | set | None = None) -> str:
    """
    Maps market to broad sleeve. All EU_QUOTAS markets map to 'europe'.
    """
    m = str(market or "").strip().lower()
    if eu_quotas is None:
        eu_markets = {
            "uk", "germany", "france", "switzerland", "netherlands",
            "sweden", "denmark", "norway", "finland", "italy",
            "spain", "belgium", "austria", "poland", "portugal", "europe"
        }
    elif isinstance(eu_quotas, dict):
        eu_markets = set(eu_quotas.keys()) | {"europe"}
    else:
        eu_markets = set(eu_quotas) | {"europe"}

    if m in eu_markets:
        return "europe"
    if m in {"america", "usa", "us"}:
        return "usa"
    if m in {"japan"}:
        return "japan"
    if m in {"korea"}:
        return "korea"
    if m in {"hongkong", "hong_kong"}:
        return "hong_kong"
    if m in {"canada", "canada_us"}:
        return "canada"
    if m in {"taiwan", "taiwan_us"}:
        return "taiwan"
    if m in {"bond"}:
        return "bond"
    return m


def build_no_canada_tsm_policy(
    groups_available: dict[str, list[str]],
    sleeve_by_ticker: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """
    Builds candidate pool and sleeve mapping for candidate scenario:
    Explicitly subtracts all tickers in canada or scanner taiwan groups except TSM,
    appends TSM exactly once, asserts candidate intersection with banned tickers is empty,
    and returns a candidate sleeve map with TSM=usa.
    """
    banned = (
        set(groups_available.get("canada", [])) | set(groups_available.get("taiwan", []))
    ) - {"TSM"}

    raw_candidates = [
        t for key in sorted(groups_available.keys())
        for t in groups_available[key]
    ]

    filtered_candidates = [t for t in raw_candidates if t not in banned and t != "TSM"]
    candidate_pool = list(dict.fromkeys(filtered_candidates))
    candidate_pool.append("TSM")

    assert candidate_pool.count("TSM") == 1, (
        f"TSM count in candidate pool is {candidate_pool.count('TSM')}, expected 1"
    )
    intersection = set(candidate_pool) & banned
    assert len(intersection) == 0, (
        f"Candidate pool intersection with banned tickers is non-empty: {intersection}"
    )

    candidate_sleeve_map = dict(sleeve_by_ticker)
    candidate_sleeve_map["TSM"] = "usa"

    return candidate_pool, candidate_sleeve_map


def compute_within_sleeve_percentiles(
    row_r63: pd.Series, sleeve_mapping: dict[str, str]
) -> pd.Series:
    """
    Calculates per-date within-sleeve percentile ranks for non-NaN momentum values.
    Returns percentile ranks in range (0, 1.0].
    """
    clean = row_r63.dropna()
    if clean.empty:
        return pd.Series(dtype=float)

    sleeves = pd.Series({t: sleeve_mapping.get(t, "unknown") for t in clean.index}, index=clean.index)
    df_tmp = pd.DataFrame({"raw": clean, "sleeve": sleeves})
    pcts = df_tmp.groupby("sleeve")["raw"].rank(pct=True)
    return pcts


compute_market_percentiles = compute_within_sleeve_percentiles


def rank_assets_df(
    row_r63: pd.Series,
    sleeve_mapping: dict[str, str],
    mode: str = "market_percentile",
) -> pd.DataFrame:
    """
    Ranks tickers for a single date based on mode ('market_percentile' or 'raw').
    Returns a DataFrame with columns: ['ticker', 'sleeve', 'r63_jpy', 'ranking_score', 'percentile']
    For 'raw' mode, reproduces ref.run_strategy sort_values(ascending=False) behavior exactly.
    For 'market_percentile' mode, sorted deterministically by (-ranking_score, -r63_jpy, ticker).
    """
    clean = row_r63.dropna()
    if clean.empty:
        return pd.DataFrame(columns=["ticker", "sleeve", "r63_jpy", "ranking_score", "percentile"])

    sleeves = {t: sleeve_mapping.get(t, "unknown") for t in clean.index}

    if mode == "raw":
        sorted_series = clean.sort_values(ascending=False)
        rows = []
        for ticker, raw_val in sorted_series.items():
            raw_v = float(raw_val)
            rows.append({
                "ticker": str(ticker),
                "sleeve": sleeves.get(ticker, "unknown"),
                "r63_jpy": raw_v,
                "ranking_score": raw_v,
                "percentile": None,
            })
        return pd.DataFrame(rows)

    pcts = compute_within_sleeve_percentiles(clean, sleeve_mapping)
    rows = []
    for ticker, raw_val in clean.items():
        raw_v = float(raw_val)
        pct_v = float(pcts.at[ticker]) if (ticker in pcts and pd.notna(pcts.at[ticker])) else None
        rows.append({
            "ticker": str(ticker),
            "sleeve": sleeves.get(ticker, "unknown"),
            "r63_jpy": raw_v,
            "ranking_score": pct_v,
            "percentile": pct_v,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["ranking_score", "r63_jpy", "ticker"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    return df


def run_strategy(
    ref: Any,
    scenario: str,
    equity_tickers: list[str],
    native_close: pd.DataFrame,
    jpy_close: pd.DataFrame,
    market_by_ticker: dict[str, str],
    sleeve_by_ticker: dict[str, str],
    use_percentile: bool = False,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """
    Executes backtest for a scenario with optional within-sleeve percentile scoring.
    Maintains unchanged defense regime and transaction cost logic.
    Deterministic sorting: (-score, -raw, ticker) in percentile mode, exact ref sorting in raw mode.
    """
    calendar = jpy_close.index
    bond_native = native_close.reindex(columns=[t for t in ref.BONDS if t in native_close.columns])
    candidates = [t for t in equity_tickers if t in jpy_close.columns]
    r63_jpy = jpy_close[candidates].pct_change(63, fill_method=None)
    r63_native = native_close[candidates].pct_change(63, fill_method=None)
    bond_r63 = bond_native.pct_change(63, fill_method=None)
    asset_returns = (
        jpy_close.reindex(columns=candidates + list(bond_native.columns))
        .pct_change(fill_method=None)
        .fillna(0.0)
    )

    rebal = set(ref.rebalance_dates(calendar))
    weights = pd.Series(dtype=float)
    daily = []
    selections = []
    regime_count = 0
    turnover_sum = 0.0

    for idx, date in enumerate(calendar):
        if idx == 0:
            daily.append(0.0)
        else:
            row_ret = asset_returns.loc[date]
            active = weights.index.intersection(row_ret.index)
            port_ret = float(
                (weights.reindex(active).fillna(0) * row_ret.reindex(active).fillna(0)).sum()
            )
            daily.append(port_ret)
            if len(weights):
                grown = weights * (1.0 + row_ret.reindex(weights.index).fillna(0.0))
                denom = float(grown.sum())
                weights = grown / denom if denom > 0 else pd.Series(dtype=float)

        if date not in rebal:
            continue

        stock_median = float(r63_native.loc[date].median(skipna=True))
        bonds_today = bond_r63.loc[date].dropna().sort_values(ascending=False)
        regime = bool(
            stock_median < 0 and len(bonds_today) and float(bonds_today.iloc[0]) > 0.08
        )

        if regime:
            best = str(bonds_today.index[0])
            target = pd.Series({best: 1.0})
            regime_count += 1
            chosen_details = [
                {
                    "ticker": best,
                    "market": "bond",
                    "sleeve": "bond",
                    "weight": 1.0,
                    "r63_jpy": float(bond_r63.at[date, best]) if best in bond_r63.columns and pd.notna(bond_r63.at[date, best]) else None,
                    "score": 1.0,
                    "percentile": None,
                }
            ]
        else:
            row_r63 = r63_jpy.loc[date].dropna()
            ranked_df = rank_assets_df(
                row_r63, sleeve_by_ticker, mode="market_percentile" if use_percentile else "raw"
            )
            top_df = ranked_df.head(ref.TOP_N)
            chosen_tickers = list(top_df["ticker"])

            if chosen_tickers:
                w = 1.0 / len(chosen_tickers)
                target = pd.Series(w, index=chosen_tickers)
            else:
                target = pd.Series(dtype=float)

            chosen_details = []
            for _, r_row in top_df.iterrows():
                tk = r_row["ticker"]
                chosen_details.append({
                    "ticker": tk,
                    "market": market_by_ticker.get(tk, "unknown"),
                    "sleeve": sleeve_by_ticker.get(tk, "unknown"),
                    "weight": float(target.get(tk, 0.0)),
                    "r63_jpy": r_row["r63_jpy"],
                    "score": r_row["ranking_score"],
                    "percentile": r_row["percentile"],
                })

        union = weights.index.union(target.index)
        one_way_turnover = 0.5 * float(
            (target.reindex(union).fillna(0) - weights.reindex(union).fillna(0)).abs().sum()
        )
        if not weights.empty or not target.empty:
            cost = ref.COST_RATE * (1.0 if weights.empty else one_way_turnover)
            daily[-1] -= cost
            turnover_sum += (1.0 if weights.empty else one_way_turnover)
        weights = target

        for detail in chosen_details:
            selections.append({
                "scenario": scenario,
                "date": date.date().isoformat(),
                "ticker": detail["ticker"],
                "market": detail["market"],
                "sleeve": detail["sleeve"],
                "weight": detail["weight"],
                "regime": regime,
                "r63_jpy": detail["r63_jpy"],
                "score": detail["score"],
                "percentile": detail["percentile"],
            })

    returns = pd.Series(daily, index=calendar, name=scenario)
    metrics = ref.calc_metrics(returns.loc[returns.index >= pd.Timestamp(ref.TRADE_START)])
    metrics.update({
        "scenario": scenario,
        "candidate_count": len(candidates),
        "rebalance_count": len(rebal),
        "regime_count": regime_count,
        "one_way_turnover_sum": turnover_sum,
    })
    return returns, pd.DataFrame(selections), metrics


def compute_diagnostics(
    ref: Any,
    selections_df: pd.DataFrame,
    jpy_close: pd.DataFrame,
    native_close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Computes 21-day forward-return diagnostics per selection slot,
    forward_return_summary grouped by scenario and sleeve, and
    displacement metrics between all5_raw and all5_market_percentile.
    Excludes incomplete final windows (sessions_held < ref.REBALANCE_STEP) from forward summaries and displacement return aggregates.
    """
    calendar = jpy_close.index
    rebal_list = ref.rebalance_dates(calendar)
    rebal_map = {}
    sessions_held_map = {}
    complete_window_map = {}

    for i, d in enumerate(rebal_list):
        d_str = d.date().isoformat()
        next_d = rebal_list[i + 1] if i + 1 < len(rebal_list) else calendar[-1]
        rebal_map[d_str] = next_d
        sess_count = int(len(calendar[(calendar > d) & (calendar <= next_d)]))
        sessions_held_map[d_str] = sess_count
        complete_window_map[d_str] = bool(sess_count >= ref.REBALANCE_STEP)

    # Forward return & MAE/MFE calculations per selection row
    fwd_returns = []
    mae_list = []
    mfe_list = []
    sessions_held_list = []
    complete_window_list = []

    for idx, row in selections_df.iterrows():
        d_str = str(row["date"])
        ticker = str(row["ticker"])
        d_start = pd.Timestamp(d_str)
        d_end = rebal_map.get(d_str, calendar[-1])
        sess_count = sessions_held_map.get(d_str, 0)
        comp_win = complete_window_map.get(d_str, False)

        sessions_held_list.append(sess_count)
        complete_window_list.append(comp_win)

        if ticker in jpy_close.columns:
            p_start = jpy_close.at[d_start, ticker] if d_start in jpy_close.index else np.nan
            p_end = jpy_close.at[d_end, ticker] if d_end in jpy_close.index else np.nan
        elif ticker in native_close.columns:
            p_start = native_close.at[d_start, ticker] if d_start in native_close.index else np.nan
            p_end = native_close.at[d_end, ticker] if d_end in native_close.index else np.nan
        else:
            p_start, p_end = np.nan, np.nan

        fwd_ret = (float(p_end) / float(p_start) - 1.0) if (pd.notna(p_start) and pd.notna(p_end) and p_start > 0) else np.nan
        fwd_returns.append(fwd_ret)

        if comp_win and ticker in jpy_close.columns and d_start in jpy_close.index and d_end in jpy_close.index:
            w_prices = jpy_close.loc[(jpy_close.index >= d_start) & (jpy_close.index <= d_end), ticker]
            if not w_prices.empty and not w_prices.isna().any():
                p0 = float(w_prices.iloc[0])
                if p0 > 0:
                    mae_val = float(w_prices.min() / p0 - 1.0)
                    mfe_val = float(w_prices.max() / p0 - 1.0)
                else:
                    mae_val, mfe_val = np.nan, np.nan
            else:
                mae_val, mfe_val = np.nan, np.nan
        else:
            mae_val, mfe_val = np.nan, np.nan

        mae_list.append(mae_val)
        mfe_list.append(mfe_val)

    fwd_df = selections_df.copy()
    fwd_df["forward_21d_return"] = fwd_returns
    fwd_df["mae"] = mae_list
    fwd_df["mfe"] = mfe_list
    fwd_df["sessions_held"] = sessions_held_list
    fwd_df["complete_window"] = complete_window_list

    # Summary grouped by scenario and broad sleeve (excluding incomplete windows)
    fwd_valid = fwd_df[fwd_df["complete_window"] & fwd_df["forward_21d_return"].notna()]
    summary_rows = []
    if not fwd_valid.empty:
        for (scenario, sleeve), group in fwd_valid.groupby(["scenario", "sleeve"]):
            rets = group["forward_21d_return"]
            summary_rows.append({
                "scenario": scenario,
                "sleeve": sleeve,
                "count": int(len(rets)),
                "mean": float(rets.mean()),
                "median": float(rets.median()),
                "win_rate": float((rets > 0).mean()),
            })
    fwd_summary_df = pd.DataFrame(summary_rows)

    # Build scenario + date + ticker lookup to guarantee no cross-scenario overwrites
    fwd_map = {}
    for idx, row in fwd_df.iterrows():
        key = (str(row["scenario"]), str(row["date"]), str(row["ticker"]))
        fwd_map[key] = row["forward_21d_return"]

    # Displacement per rebalance date
    displacement_rows = []
    raw_sel = selections_df[selections_df["scenario"] == "all5_raw"]
    pct_sel = selections_df[selections_df["scenario"] == "all5_market_percentile"]

    for d in rebal_list:
        d_str = d.date().isoformat()
        raw_sub = raw_sel[raw_sel["date"] == d_str]
        pct_sub = pct_sel[pct_sel["date"] == d_str]

        if raw_sub.empty and pct_sub.empty:
            continue

        regime = bool(raw_sub["regime"].iloc[0]) if not raw_sub.empty else False
        raw_tickers = set(raw_sub["ticker"])
        pct_tickers = set(pct_sub["ticker"])

        overlap = raw_tickers & pct_tickers
        displaced = raw_tickers - pct_tickers
        added = pct_tickers - raw_tickers

        union_set = raw_tickers | pct_tickers
        jaccard = len(overlap) / float(len(union_set)) if union_set else 1.0

        sess_count = sessions_held_map.get(d_str, 0)
        comp_win = complete_window_map.get(d_str, False)

        raw_fwds = [fwd_map[("all5_raw", d_str, t)] for t in raw_tickers if ("all5_raw", d_str, t) in fwd_map and pd.notna(fwd_map[("all5_raw", d_str, t)])]
        pct_fwds = [fwd_map[("all5_market_percentile", d_str, t)] for t in pct_tickers if ("all5_market_percentile", d_str, t) in fwd_map and pd.notna(fwd_map[("all5_market_percentile", d_str, t)])]
        overlap_fwds = [fwd_map[("all5_raw", d_str, t)] for t in overlap if ("all5_raw", d_str, t) in fwd_map and pd.notna(fwd_map[("all5_raw", d_str, t)])]
        displaced_fwds = [fwd_map[("all5_raw", d_str, t)] for t in displaced if ("all5_raw", d_str, t) in fwd_map and pd.notna(fwd_map[("all5_raw", d_str, t)])]
        added_fwds = [fwd_map[("all5_market_percentile", d_str, t)] for t in added if ("all5_market_percentile", d_str, t) in fwd_map and pd.notna(fwd_map[("all5_market_percentile", d_str, t)])]

        fwd_raw_mean = float(np.mean(raw_fwds)) if raw_fwds else np.nan
        fwd_pct_mean = float(np.mean(pct_fwds)) if pct_fwds else np.nan
        fwd_overlap_mean = float(np.mean(overlap_fwds)) if overlap_fwds else np.nan
        fwd_displaced_mean = float(np.mean(displaced_fwds)) if displaced_fwds else np.nan
        fwd_added_mean = float(np.mean(added_fwds)) if added_fwds else np.nan
        fwd_delta = (fwd_pct_mean - fwd_raw_mean) if (pd.notna(fwd_pct_mean) and pd.notna(fwd_raw_mean)) else np.nan

        displacement_rows.append({
            "date": d_str,
            "next_rebalance_date": rebal_map.get(d_str, calendar[-1]).date().isoformat(),
            "sessions_held": sess_count,
            "complete_window": comp_win,
            "regime": regime,
            "raw_count": len(raw_tickers),
            "pct_count": len(pct_tickers),
            "overlap_count": len(overlap),
            "displaced_count": len(displaced),
            "added_count": len(added),
            "jaccard_similarity": jaccard,
            "fwd_ret_raw_pct": fwd_raw_mean * 100.0 if pd.notna(fwd_raw_mean) else None,
            "fwd_ret_pct_pct": fwd_pct_mean * 100.0 if pd.notna(fwd_pct_mean) else None,
            "fwd_ret_overlap_pct": fwd_overlap_mean * 100.0 if pd.notna(fwd_overlap_mean) else None,
            "fwd_ret_displaced_pct": fwd_displaced_mean * 100.0 if pd.notna(fwd_displaced_mean) else None,
            "fwd_ret_added_pct": fwd_added_mean * 100.0 if pd.notna(fwd_added_mean) else None,
            "fwd_ret_delta_pct": fwd_delta * 100.0 if pd.notna(fwd_delta) else None,
        })

    disp_df = pd.DataFrame(displacement_rows)

    complete_non_regime_disp = disp_df[~disp_df["regime"] & disp_df["complete_window"]] if not disp_df.empty else pd.DataFrame()
    non_regime_disp = disp_df[~disp_df["regime"]] if not disp_df.empty else pd.DataFrame()

    disp_summary = {
        "total_rebalance_dates": int(len(disp_df)),
        "non_regime_rebalance_dates": int(len(non_regime_disp)),
        "complete_non_regime_rebalance_dates": int(len(complete_non_regime_disp)),
        "avg_overlap_count": float(non_regime_disp["overlap_count"].mean()) if not non_regime_disp.empty else np.nan,
        "avg_displaced_count": float(non_regime_disp["displaced_count"].mean()) if not non_regime_disp.empty else np.nan,
        "avg_jaccard_similarity": float(non_regime_disp["jaccard_similarity"].mean()) if not non_regime_disp.empty else np.nan,
        "avg_fwd_ret_raw_pct": float(complete_non_regime_disp["fwd_ret_raw_pct"].mean()) if not complete_non_regime_disp.empty else np.nan,
        "avg_fwd_ret_pct_pct": float(complete_non_regime_disp["fwd_ret_pct_pct"].mean()) if not complete_non_regime_disp.empty else np.nan,
        "avg_fwd_ret_displaced_pct": float(complete_non_regime_disp["fwd_ret_displaced_pct"].mean()) if not complete_non_regime_disp.empty else np.nan,
        "avg_fwd_ret_added_pct": float(complete_non_regime_disp["fwd_ret_added_pct"].mean()) if not complete_non_regime_disp.empty else np.nan,
        "avg_fwd_ret_delta_pct": float(complete_non_regime_disp["fwd_ret_delta_pct"].mean()) if not complete_non_regime_disp.empty else np.nan,
    }

    return fwd_df, fwd_summary_df, disp_df, disp_summary


def compute_policy_delta_diagnostics(
    ref: Any,
    selections_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    jpy_close: pd.DataFrame,
    native_close: pd.DataFrame,
    baseline_scenario: str = "all5_market_percentile",
    candidate_scenario: str = "no_canada_tsm_usa_sleeve_market_percentile",
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Computes policy-delta diagnostics comparing baseline_scenario to candidate_scenario.
    Returns:
      - policy_delta_df (pd.DataFrame): per-rebalance date policy-delta metrics.
      - policy_delta_summary (dict): aggregate summary statistics for JSON export.
      - tsm_selection_df (pd.DataFrame): explicit TSM selection slot diagnostics.
    """
    calendar = jpy_close.index
    rebal_list = ref.rebalance_dates(calendar)
    rebal_map = {}
    sessions_held_map = {}
    complete_window_map = {}

    for i, d in enumerate(rebal_list):
        d_str = d.date().isoformat()
        next_d = rebal_list[i + 1] if i + 1 < len(rebal_list) else calendar[-1]
        rebal_map[d_str] = next_d
        sess_count = int(len(calendar[(calendar > d) & (calendar <= next_d)]))
        sessions_held_map[d_str] = sess_count
        complete_window_map[d_str] = bool(sess_count >= ref.REBALANCE_STEP)

    fwd_map = {}
    mae_map = {}
    mfe_map = {}
    for idx, row in fwd_df.iterrows():
        key = (str(row["scenario"]), str(row["date"]), str(row["ticker"]))
        fwd_map[key] = row["forward_21d_return"]
        if "mae" in row and pd.notna(row["mae"]):
            mae_map[key] = row["mae"]
        if "mfe" in row and pd.notna(row["mfe"]):
            mfe_map[key] = row["mfe"]

    policy_delta_rows = []
    base_sel = selections_df[selections_df["scenario"] == baseline_scenario]
    cand_sel = selections_df[selections_df["scenario"] == candidate_scenario]

    for d in rebal_list:
        d_str = d.date().isoformat()
        b_sub = base_sel[base_sel["date"] == d_str]
        c_sub = cand_sel[cand_sel["date"] == d_str]

        if b_sub.empty and c_sub.empty:
            continue

        b_regime = bool(b_sub["regime"].iloc[0]) if not b_sub.empty else False
        c_regime = bool(c_sub["regime"].iloc[0]) if not c_sub.empty else False
        regime_mismatch = bool(b_regime != c_regime)
        regime = bool(b_regime or c_regime)

        b_tickers = set(b_sub["ticker"])
        c_tickers = set(c_sub["ticker"])

        overlap = b_tickers & c_tickers
        dropped = b_tickers - c_tickers
        added = c_tickers - b_tickers

        union_set = b_tickers | c_tickers
        jaccard = len(overlap) / float(len(union_set)) if union_set else 1.0

        sess_count = sessions_held_map.get(d_str, 0)
        comp_win = complete_window_map.get(d_str, False)

        b_fwds = [fwd_map[(baseline_scenario, d_str, t)] for t in sorted(b_tickers) if (baseline_scenario, d_str, t) in fwd_map and pd.notna(fwd_map[(baseline_scenario, d_str, t)])]
        c_fwds = [fwd_map[(candidate_scenario, d_str, t)] for t in sorted(c_tickers) if (candidate_scenario, d_str, t) in fwd_map and pd.notna(fwd_map[(candidate_scenario, d_str, t)])]
        overlap_fwds = [fwd_map[(baseline_scenario, d_str, t)] for t in sorted(overlap) if (baseline_scenario, d_str, t) in fwd_map and pd.notna(fwd_map[(baseline_scenario, d_str, t)])]
        dropped_fwds = [fwd_map[(baseline_scenario, d_str, t)] for t in sorted(dropped) if (baseline_scenario, d_str, t) in fwd_map and pd.notna(fwd_map[(baseline_scenario, d_str, t)])]
        added_fwds = [fwd_map[(candidate_scenario, d_str, t)] for t in sorted(added) if (candidate_scenario, d_str, t) in fwd_map and pd.notna(fwd_map[(candidate_scenario, d_str, t)])]

        fwd_b_mean = float(np.mean(b_fwds)) if b_fwds else np.nan
        fwd_c_mean = float(np.mean(c_fwds)) if c_fwds else np.nan
        fwd_overlap_mean = float(np.mean(overlap_fwds)) if overlap_fwds else np.nan
        fwd_dropped_mean = float(np.mean(dropped_fwds)) if dropped_fwds else np.nan
        fwd_added_mean = float(np.mean(added_fwds)) if added_fwds else np.nan
        fwd_delta = (fwd_c_mean - fwd_b_mean) if (pd.notna(fwd_c_mean) and pd.notna(fwd_b_mean)) else np.nan

        canada_dropped = [
            t for t in sorted(dropped)
            if (not b_sub.empty and (
                b_sub.loc[b_sub["ticker"] == t, "sleeve"].iloc[0] == "canada" or
                b_sub.loc[b_sub["ticker"] == t, "market"].iloc[0] in {"canada", "canada_us"}
            ))
        ]

        taiwan_dropped = [
            t for t in sorted(dropped)
            if t != "TSM" and (not b_sub.empty and (
                b_sub.loc[b_sub["ticker"] == t, "sleeve"].iloc[0] == "taiwan" or
                b_sub.loc[b_sub["ticker"] == t, "market"].iloc[0] in {"taiwan", "taiwan_us"}
            ))
        ]

        tsm_selected = "TSM" in c_tickers
        tsm_row = c_sub[c_sub["ticker"] == "TSM"] if tsm_selected else pd.DataFrame()
        tsm_r63 = float(tsm_row["r63_jpy"].iloc[0]) if (tsm_selected and not tsm_row.empty and pd.notna(tsm_row["r63_jpy"].iloc[0])) else np.nan
        tsm_fwd = fwd_map.get((candidate_scenario, d_str, "TSM"), np.nan)

        policy_delta_rows.append({
            "review_only": True,
            "not_approved_for_live_trading": True,
            "live_order_enabled": False,
            "status": "REFERENCE_ONLY",
            "date": d_str,
            "next_rebalance_date": rebal_map.get(d_str, calendar[-1]).date().isoformat(),
            "sessions_held": sess_count,
            "complete_window": comp_win,
            "regime": regime,
            "baseline_regime": b_regime,
            "candidate_regime": c_regime,
            "regime_mismatch": regime_mismatch,
            "baseline_count": len(b_tickers),
            "candidate_count": len(c_tickers),
            "overlap_count": len(overlap),
            "dropped_count": len(dropped),
            "added_count": len(added),
            "jaccard_similarity": jaccard,
            "fwd_ret_baseline_pct": fwd_b_mean * 100.0 if pd.notna(fwd_b_mean) else None,
            "fwd_ret_candidate_pct": fwd_c_mean * 100.0 if pd.notna(fwd_c_mean) else None,
            "fwd_ret_overlap_pct": fwd_overlap_mean * 100.0 if pd.notna(fwd_overlap_mean) else None,
            "fwd_ret_dropped_pct": fwd_dropped_mean * 100.0 if pd.notna(fwd_dropped_mean) else None,
            "fwd_ret_added_pct": fwd_added_mean * 100.0 if pd.notna(fwd_added_mean) else None,
            "fwd_ret_delta_pct": fwd_delta * 100.0 if pd.notna(fwd_delta) else None,
            "canada_dropped_count": len(canada_dropped),
            "taiwan_dropped_count": len(taiwan_dropped),
            "tsm_selected": tsm_selected,
            "tsm_r63_jpy": tsm_r63 if pd.notna(tsm_r63) else None,
            "tsm_forward_21d_return": tsm_fwd if pd.notna(tsm_fwd) else None,
        })

    delta_df = pd.DataFrame(policy_delta_rows)

    tsm_selection_df = fwd_df[(fwd_df["scenario"] == candidate_scenario) & (fwd_df["ticker"] == "TSM")].copy()
    tsm_selection_df["review_only"] = True
    tsm_selection_df["not_approved_for_live_trading"] = True
    tsm_selection_df["live_order_enabled"] = False
    tsm_selection_df["status"] = "REFERENCE_ONLY"

    both_non_regime_delta = delta_df[~delta_df["baseline_regime"] & ~delta_df["candidate_regime"]] if not delta_df.empty else pd.DataFrame()
    comp_both_non_regime_delta = delta_df[~delta_df["baseline_regime"] & ~delta_df["candidate_regime"] & delta_df["complete_window"]] if not delta_df.empty else pd.DataFrame()

    mismatch_df = delta_df[delta_df["regime_mismatch"]] if not delta_df.empty else pd.DataFrame()
    regime_mismatch_count = len(mismatch_df)
    regime_mismatch_dates = sorted(list(mismatch_df["date"])) if not mismatch_df.empty else []

    all_dropped_fwds = []
    all_dropped_maes = []
    all_dropped_mfes = []
    all_added_fwds = []
    all_added_maes = []
    all_added_mfes = []

    for idx, row in comp_both_non_regime_delta.iterrows():
        d_str = str(row["date"])
        b_sub = base_sel[base_sel["date"] == d_str]
        c_sub = cand_sel[cand_sel["date"] == d_str]
        b_tickers = set(b_sub["ticker"])
        c_tickers = set(c_sub["ticker"])
        dropped = b_tickers - c_tickers
        added = c_tickers - b_tickers
        for t in sorted(dropped):
            val = fwd_map.get((baseline_scenario, d_str, t))
            m_val = mae_map.get((baseline_scenario, d_str, t))
            fe_val = mfe_map.get((baseline_scenario, d_str, t))
            if pd.notna(val):
                all_dropped_fwds.append(val)
            if m_val is not None and pd.notna(m_val):
                all_dropped_maes.append(m_val)
            if fe_val is not None and pd.notna(fe_val):
                all_dropped_mfes.append(fe_val)
        for t in sorted(added):
            val = fwd_map.get((candidate_scenario, d_str, t))
            m_val = mae_map.get((candidate_scenario, d_str, t))
            fe_val = mfe_map.get((candidate_scenario, d_str, t))
            if pd.notna(val):
                all_added_fwds.append(val)
            if m_val is not None and pd.notna(m_val):
                all_added_maes.append(m_val)
            if fe_val is not None and pd.notna(fe_val):
                all_added_mfes.append(fe_val)

    def _calc_extended_stats(
        fwds: list[float], maes: list[float], mfes: list[float] | None = None
    ) -> dict[str, Any]:
        valid_fwds = [float(f) for f in fwds if pd.notna(f)]
        valid_maes = [float(m) for m in maes if pd.notna(m)]
        valid_mfes = [float(m) for m in (mfes or []) if pd.notna(m)]

        count = len(valid_fwds)
        if count == 0:
            return {
                "weighting": "unweighted_slot_level",
                "return_unit": "decimal_fraction",
                "count": 0,
                "mean": None,
                "mean_forward_return": None,
                "mean_forward_return_fraction": None,
                "expectancy": None,
                "expectancy_fraction": None,
                "median": None,
                "median_forward_return": None,
                "median_forward_return_fraction": None,
                "win_rate": None,
                "win_rate_fraction": None,
                "profit_factor": None,
                "profit_factor_state": "no_data",
                "mae_count": 0,
                "mean_mae": None,
                "mean_mae_fraction": None,
                "mfe_count": 0,
                "mean_mfe": None,
                "mean_mfe_fraction": None,
            }

        mean_val = float(np.mean(valid_fwds))
        median_val = float(np.median(valid_fwds))
        win_rate_val = float((np.array(valid_fwds) > 0).mean())

        pos_sum = float(np.sum([r for r in valid_fwds if r > 0]))
        neg_sum = float(np.sum([r for r in valid_fwds if r < 0]))

        if pos_sum == 0 and neg_sum == 0:
            profit_factor = None
            profit_factor_state = "all_zero"
        elif neg_sum == 0:
            profit_factor = None
            profit_factor_state = "no_losing_trades"
        elif abs(neg_sum) > 0:
            profit_factor = float(pos_sum / abs(neg_sum))
            profit_factor_state = "finite"
        else:
            profit_factor = None
            profit_factor_state = "no_data"

        mean_mae = float(np.mean(valid_maes)) if valid_maes else None
        mean_mfe = float(np.mean(valid_mfes)) if valid_mfes else None

        return {
            "weighting": "unweighted_slot_level",
            "return_unit": "decimal_fraction",
            "count": count,
            "mean": mean_val,
            "mean_forward_return": mean_val,
            "mean_forward_return_fraction": mean_val,
            "expectancy": mean_val,
            "expectancy_fraction": mean_val,
            "median": median_val,
            "median_forward_return": median_val,
            "median_forward_return_fraction": median_val,
            "win_rate": win_rate_val,
            "win_rate_fraction": win_rate_val,
            "profit_factor": profit_factor,
            "profit_factor_state": profit_factor_state,
            "mae_count": len(valid_maes),
            "mean_mae": mean_mae,
            "mean_mae_fraction": mean_mae,
            "mfe_count": len(valid_mfes),
            "mean_mfe": mean_mfe,
            "mean_mfe_fraction": mean_mfe,
        }

    dropped_stats = _calc_extended_stats(all_dropped_fwds, all_dropped_maes, all_dropped_mfes)
    added_stats = _calc_extended_stats(all_added_fwds, all_added_maes, all_added_mfes)

    valid_dates_set = set(comp_both_non_regime_delta["date"]) if not comp_both_non_regime_delta.empty else set()

    canada_base_fwd = fwd_df[
        (fwd_df["scenario"] == baseline_scenario) &
        ((fwd_df["sleeve"] == "canada") | (fwd_df["market"].isin(["canada", "canada_us"])))
    ]
    comp_canada = canada_base_fwd[
        canada_base_fwd["date"].isin(valid_dates_set) &
        canada_base_fwd["complete_window"] &
        canada_base_fwd["forward_21d_return"].notna()
    ]
    can_rets = comp_canada["forward_21d_return"].tolist()
    can_maes = comp_canada["mae"].dropna().tolist() if "mae" in comp_canada else []
    can_mfes = comp_canada["mfe"].dropna().tolist() if "mfe" in comp_canada else []
    can_ext = _calc_extended_stats(can_rets, can_maes, can_mfes)

    canada_stats = {
        "total_baseline_canada_selections": len(canada_base_fwd),
        "complete_window_canada_selections": len(comp_canada),
        "unique_canada_tickers_displaced": sorted(list(set(canada_base_fwd["ticker"]))),
        **can_ext,
    }

    taiwan_base_fwd = fwd_df[
        (fwd_df["scenario"] == baseline_scenario) &
        (fwd_df["ticker"] != "TSM") &
        ((fwd_df["sleeve"] == "taiwan") | (fwd_df["market"].isin(["taiwan", "taiwan_us"])))
    ]
    comp_taiwan = taiwan_base_fwd[
        taiwan_base_fwd["date"].isin(valid_dates_set) &
        taiwan_base_fwd["complete_window"] &
        taiwan_base_fwd["forward_21d_return"].notna()
    ]
    taiwan_rets = comp_taiwan["forward_21d_return"].tolist()
    taiwan_maes = comp_taiwan["mae"].dropna().tolist() if "mae" in comp_taiwan else []
    taiwan_mfes = comp_taiwan["mfe"].dropna().tolist() if "mfe" in comp_taiwan else []
    taiwan_ext = _calc_extended_stats(taiwan_rets, taiwan_maes, taiwan_mfes)

    taiwan_stats = {
        "total_baseline_taiwan_selections": len(taiwan_base_fwd),
        "complete_window_taiwan_selections": len(comp_taiwan),
        "unique_taiwan_tickers_displaced": sorted(list(set(taiwan_base_fwd["ticker"]))),
        **taiwan_ext,
    }

    comp_tsm = tsm_selection_df[
        tsm_selection_df["date"].isin(valid_dates_set) &
        tsm_selection_df["complete_window"] &
        tsm_selection_df["forward_21d_return"].notna()
    ]
    tsm_rets = comp_tsm["forward_21d_return"].tolist()
    tsm_maes = comp_tsm["mae"].dropna().tolist() if "mae" in comp_tsm else []
    tsm_mfes = comp_tsm["mfe"].dropna().tolist() if "mfe" in comp_tsm else []
    tsm_ext = _calc_extended_stats(tsm_rets, tsm_maes, tsm_mfes)

    tsm_stats = {
        "total_selections": len(tsm_selection_df),
        "complete_window_selections": len(comp_tsm),
        **tsm_ext,
    }

    summary = {
        "review_only": True,
        "not_approved_for_live_trading": True,
        "live_order_enabled": False,
        "status": "REFERENCE_ONLY",
        "baseline_scenario": baseline_scenario,
        "candidate_scenario": candidate_scenario,
        "policy_declaration": "Exclude every Canada-US candidate and scanner-derived Taiwan candidate; add explicit TSM ADR (ticker='TSM', market='taiwan_us', currency='USD', source_symbol='NYSE:TSM'); TSM market taiwan_us with ranking sleeve usa; percentile mode.",
        "rebalance_counts": {
            "total_rebalance_dates": len(delta_df),
            "non_regime_rebalance_dates": len(both_non_regime_delta),
            "complete_non_regime_rebalance_dates": len(comp_both_non_regime_delta),
            "regime_mismatch_count": regime_mismatch_count,
            "regime_mismatch_dates": regime_mismatch_dates,
        },
        "average_metrics_non_regime": {
            "avg_overlap_count": float(both_non_regime_delta["overlap_count"].mean()) if not both_non_regime_delta.empty else None,
            "avg_dropped_count": float(both_non_regime_delta["dropped_count"].mean()) if not both_non_regime_delta.empty else None,
            "avg_added_count": float(both_non_regime_delta["added_count"].mean()) if not both_non_regime_delta.empty else None,
            "avg_jaccard_similarity": float(both_non_regime_delta["jaccard_similarity"].mean()) if not both_non_regime_delta.empty else None,
            "avg_fwd_ret_baseline_pct": float(comp_both_non_regime_delta["fwd_ret_baseline_pct"].mean()) if not comp_both_non_regime_delta.empty else None,
            "avg_fwd_ret_candidate_pct": float(comp_both_non_regime_delta["fwd_ret_candidate_pct"].mean()) if not comp_both_non_regime_delta.empty else None,
            "avg_fwd_ret_overlap_pct": float(comp_both_non_regime_delta["fwd_ret_overlap_pct"].mean()) if not comp_both_non_regime_delta.empty else None,
            "avg_fwd_ret_dropped_pct": float(comp_both_non_regime_delta["fwd_ret_dropped_pct"].mean()) if not comp_both_non_regime_delta.empty else None,
            "avg_fwd_ret_added_pct": float(comp_both_non_regime_delta["fwd_ret_added_pct"].mean()) if not comp_both_non_regime_delta.empty else None,
            "avg_fwd_ret_delta_pct": float(comp_both_non_regime_delta["fwd_ret_delta_pct"].mean()) if not comp_both_non_regime_delta.empty else None,
        },
        "dropped_selection_statistics": dropped_stats,
        "added_selection_statistics": added_stats,
        "canada_removed_selection_statistics": canada_stats,
        "taiwan_removed_selection_statistics": taiwan_stats,
        "tsm_selection_statistics": tsm_stats,
    }

    return delta_df, summary, tsm_selection_df


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN and Infinity floats to None for valid strict JSON."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    return obj


def main() -> None:
    ref = _load_reference()

    ref_log_messages: list[str] = []
    original_log = ref.log

    def custom_log(msg: str) -> None:
        original_log(msg)
        ref_log_messages.append(str(msg))

    ref.log = custom_log

    outdir = Path(os.environ.get("OUTDIR", "global_r63_results"))
    outdir.mkdir(parents=True, exist_ok=True)

    ref.log("Building universes (single pipeline)")
    sp500 = ref.get_sp500()
    jpx400 = ref.get_jpx400()
    korea = ref.get_korea()
    europe = ref.get_europe()
    hongkong = ref.get_hongkong()
    canada, taiwan = ref.get_us_listed_foreign()

    tsm_asset = ref.Asset("TSM", "taiwan_us", "USD", "NYSE:TSM", "Taiwan Semiconductor Manufacturing Co Ltd ADR")

    groups = {
        "base": ref.dedupe_assets(sp500 + jpx400),
        "korea": ref.dedupe_assets(korea),
        "europe": ref.dedupe_assets(europe),
        "hong_kong": ref.dedupe_assets(hongkong),
        "canada": ref.dedupe_assets(canada),
        "taiwan": ref.dedupe_assets(taiwan),
    }
    universe_counts_raw = {k: len(v) for k, v in groups.items()}
    ref.log(f"Raw counts: {universe_counts_raw}")

    all_assets = ref.dedupe_assets(
        [tsm_asset]
        + [a for key in sorted(groups.keys()) for a in groups[key]]
        + [ref.Asset(t, "bond", "USD", t, t) for t in ref.BONDS]
    )
    tsm_resolved = next((a for a in all_assets if a.ticker == "TSM"), None)
    assert tsm_resolved is not None and tsm_resolved.source_symbol == "NYSE:TSM", (
        f"Resolved TSM source_symbol is {getattr(tsm_resolved, 'source_symbol', None)}, expected 'NYSE:TSM'"
    )

    assets_by_ticker = {a.ticker: a for a in all_assets}
    market_by_ticker = {a.ticker: a.market for a in all_assets}
    sleeve_by_ticker = {a.ticker: get_broad_sleeve(a.market, ref.EU_QUOTAS) for a in all_assets}

    tickers = sorted(list(assets_by_ticker.keys()))
    native = ref.download_closes(tickers)
    native = native.reindex(columns=[t for t in tickers if t in native.columns])
    native, exclusions = ref.quality_filter(native)

    calendar = native["SPY"].dropna().index if "SPY" in native else native.index
    calendar = calendar[(calendar >= pd.Timestamp(ref.START)) & (calendar < pd.Timestamp(ref.END))]
    native = native.reindex(calendar).ffill(limit=5)

    fx = ref.load_fx({assets_by_ticker[t].currency for t in native.columns}, calendar)
    jpy = ref.to_jpy(native, assets_by_ticker, fx).reindex(calendar).ffill(limit=5)
    jpy, jpy_exclusions = ref.quality_filter(jpy)
    exclusions.update({f"JPY:{k}": v for k, v in jpy_exclusions.items()})
    native = native.reindex(columns=jpy.columns)

    available = set(jpy.columns)
    if "TSM" not in available:
        raise ValueError("TSM is unavailable after quality filtering")

    groups_available = {
        key: [a.ticker for a in values if a.ticker in available]
        for key, values in sorted(groups.items())
    }

    all5_pool = list(dict.fromkeys(
        groups_available["base"] + groups_available["korea"] + groups_available["europe"]
        + groups_available["hong_kong"] + groups_available["canada"] + groups_available["taiwan"]
    ))

    no_canada_tsm_pool, candidate_sleeve_by_ticker = build_no_canada_tsm_policy(
        groups_available, sleeve_by_ticker
    )

    scenarios_spec = [
        ("base_raw", groups_available["base"], False, sleeve_by_ticker),
        ("all5_raw", all5_pool, False, sleeve_by_ticker),
        ("all5_market_percentile", all5_pool, True, sleeve_by_ticker),
        ("no_canada_tsm_usa_sleeve_market_percentile", no_canada_tsm_pool, True, candidate_sleeve_by_ticker),
    ]

    all_returns = {}
    all_selections = []
    metrics_rows = []

    for scenario_name, scenario_tickers, use_pct, scenario_sleeves in scenarios_spec:
        ref.log(f"Backtest {scenario_name}: {len(scenario_tickers)} candidates (use_percentile={use_pct})")
        returns, selections, metrics = run_strategy(
            ref, scenario_name, scenario_tickers, native, jpy,
            market_by_ticker, scenario_sleeves, use_percentile=use_pct
        )
        ew = ref.equal_weight_daily_returns(scenario_tickers, jpy)
        ew_metrics = ref.calc_metrics(ew.loc[ew.index >= pd.Timestamp(ref.TRADE_START)])
        metrics.update({f"ew_{k}": v for k, v in ew_metrics.items() if k != "days"})
        metrics.update({
            "total_return_minus_ew": metrics["total_return"] - ew_metrics["total_return"] if pd.notna(metrics["total_return"]) and pd.notna(ew_metrics["total_return"]) else np.nan,
            "cagr_minus_ew": metrics["cagr"] - ew_metrics["cagr"] if pd.notna(metrics["cagr"]) and pd.notna(ew_metrics["cagr"]) else np.nan,
            "sharpe_minus_ew": metrics["sharpe_rf0"] - ew_metrics["sharpe_rf0"] if pd.notna(metrics["sharpe_rf0"]) and pd.notna(ew_metrics["sharpe_rf0"]) else np.nan,
        })

        all_returns[scenario_name] = returns
        all_selections.append(selections)
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows).set_index("scenario")
    returns_df = pd.DataFrame(all_returns)
    selections_df = pd.concat(all_selections, ignore_index=True) if all_selections else pd.DataFrame()
    selection_counts = (
        selections_df.groupby(["scenario", "market", "sleeve"]).size().rename("selection_slots").reset_index()
        if not selections_df.empty else pd.DataFrame()
    )

    # Verification run for baseline parity against reference implementation
    ref_returns, ref_selections, _ = ref.run_strategy(
        "all5_raw_ref_verification", all5_pool, native, jpy, market_by_ticker
    )
    raw_returns = all_returns["all5_raw"]
    raw_selections = selections_df[selections_df["scenario"] == "all5_raw"]

    max_abs_daily_return_diff = float((ref_returns - raw_returns).abs().max())
    assert max_abs_daily_return_diff <= 1e-12, (
        f"Max absolute daily-return difference {max_abs_daily_return_diff} exceeds 1e-12"
    )

    ref_sel_by_date = ref_selections.groupby("date")["ticker"].apply(set)
    raw_sel_by_date = raw_selections.groupby("date")["ticker"].apply(set)
    all_dates = set(ref_sel_by_date.index) | set(raw_sel_by_date.index)
    mismatch_dates = 0
    for d in all_dates:
        t_ref = ref_sel_by_date.get(d, set())
        t_raw = raw_sel_by_date.get(d, set())
        if t_ref != t_raw:
            mismatch_dates += 1

    assert mismatch_dates == 0, (
        f"Raw selected ticker-set mismatch dates count is {mismatch_dates}, expected 0"
    )

    raw_baseline_parity = {
        "max_abs_daily_return_diff": max_abs_daily_return_diff,
        "raw_selected_ticker_set_mismatch_dates": mismatch_dates,
        "parity_verified": True,
    }

    fwd_df, fwd_summary_df, disp_df, disp_summary = compute_diagnostics(ref, selections_df, jpy, native)

    policy_delta_df, policy_delta_summary, tsm_selection_df = compute_policy_delta_diagnostics(
        ref, selections_df, fwd_df, jpy, native
    )

    b_scen = "all5_market_percentile"
    c_scen = "no_canada_tsm_usa_sleeve_market_percentile"

    port_comp_specs = [
        ("total_return", "total_return"),
        ("CAGR", "cagr"),
        ("Sharpe", "sharpe_rf0"),
        ("max_drawdown", "max_drawdown"),
        ("turnover", "one_way_turnover_sum"),
        ("EW-relative Sharpe", "sharpe_minus_ew"),
    ]

    portfolio_comparison = {}
    portfolio_comp_rows = []
    for disp_name, key in port_comp_specs:
        b_val = float(metrics_df.loc[b_scen, key]) if key in metrics_df.columns else np.nan
        c_val = float(metrics_df.loc[c_scen, key]) if key in metrics_df.columns else np.nan
        d_val = (c_val - b_val) if (pd.notna(c_val) and pd.notna(b_val)) else np.nan

        portfolio_comparison[disp_name] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta": d_val,
        }
        portfolio_comp_rows.append({
            "review_only": True,
            "not_approved_for_live_trading": True,
            "live_order_enabled": False,
            "status": "REFERENCE_ONLY",
            "metric": disp_name,
            "baseline": b_val,
            "candidate": c_val,
            "delta": d_val,
        })

    portfolio_comparison_df = pd.DataFrame(portfolio_comp_rows)
    policy_delta_summary["portfolio_comparison"] = portfolio_comparison

    # Segment metrics for 2017-2020, 2021-2023, 2024-2026
    segments = [
        ("2017-2020", ref.TRADE_START, "2020-12-31"),
        ("2021-2023", "2021-01-01", "2023-12-31"),
        ("2024-2026", "2024-01-01", "2026-12-31"),
    ]
    segment_rows = []
    for scenario_name in all_returns:
        ret_series = all_returns[scenario_name]
        scenario_tickers = [spec[1] for spec in scenarios_spec if spec[0] == scenario_name][0]
        ew_series = ref.equal_weight_daily_returns(scenario_tickers, jpy)
        for seg_name, s_start, s_end in segments:
            sub_ret = ret_series.loc[(ret_series.index >= pd.Timestamp(s_start)) & (ret_series.index <= pd.Timestamp(s_end))]
            sub_ew = ew_series.loc[(ew_series.index >= pd.Timestamp(s_start)) & (ew_series.index <= pd.Timestamp(s_end))]
            if len(sub_ret) == 0:
                continue
            m = ref.calc_metrics(sub_ret)
            ew_m = ref.calc_metrics(sub_ew)
            m_row = {
                "scenario": scenario_name,
                "segment": seg_name,
                "segment_start": sub_ret.index.min().date().isoformat(),
                "segment_end": sub_ret.index.max().date().isoformat(),
                **m,
                **{f"ew_{k}": v for k, v in ew_m.items() if k != "days"},
                "total_return_minus_ew": m["total_return"] - ew_m["total_return"] if pd.notna(m["total_return"]) and pd.notna(ew_m["total_return"]) else np.nan,
                "cagr_minus_ew": m["cagr"] - ew_m["cagr"] if pd.notna(m["cagr"]) and pd.notna(ew_m["cagr"]) else np.nan,
                "sharpe_minus_ew": m["sharpe_rf0"] - ew_m["sharpe_rf0"] if pd.notna(m["sharpe_rf0"]) and pd.notna(ew_m["sharpe_rf0"]) else np.nan,
            }
            segment_rows.append(m_row)

    segment_metrics_df = pd.DataFrame(segment_rows)

    # Save outputs
    metrics_df.to_csv(outdir / "metrics.csv")
    segment_metrics_df.to_csv(outdir / "segment_metrics.csv", index=False)
    returns_df.to_csv(outdir / "daily_returns.csv")
    selections_df.to_csv(outdir / "selections.csv", index=False)
    selection_counts.to_csv(outdir / "selection_market_counts.csv", index=False)
    fwd_df.to_csv(outdir / "forward_return_diagnostics.csv", index=False)
    fwd_summary_df.to_csv(outdir / "forward_return_summary.csv", index=False)
    disp_df.to_csv(outdir / "displacement_diagnostics.csv", index=False)
    policy_delta_df.to_csv(outdir / "policy_delta_diagnostics.csv", index=False)
    tsm_selection_df.to_csv(outdir / "tsm_selection_diagnostics.csv", index=False)
    portfolio_comparison_df.to_csv(outdir / "portfolio_comparison.csv", index=False)

    sanitized_delta_summary = sanitize_for_json(policy_delta_summary)
    (outdir / "policy_delta_summary.json").write_text(
        json.dumps(sanitized_delta_summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    coverage_df = pd.DataFrame([
        {
            "ticker": a.ticker,
            "market": a.market,
            "sleeve": sleeve_by_ticker.get(a.ticker, "unknown"),
            "candidate_ranking_sleeve": candidate_sleeve_by_ticker.get(a.ticker, sleeve_by_ticker.get(a.ticker, "unknown")),
            "currency": a.currency,
            "source_symbol": a.source_symbol,
            "name": a.name,
            "downloaded": a.ticker in native.columns,
            "quality_exclusion": exclusions.get(a.ticker),
        }
        for a in all_assets
    ])
    coverage_df.to_csv(outdir / "universe_coverage.csv", index=False)

    universe_source_warnings = []
    for msg in ref_log_messages:
        msg_lower = msg.lower()
        if any(w in msg_lower for w in ["failed", "proxy", "too small", "missing fx", "discontinuity"]):
            universe_source_warnings.append(msg)
        else:
            m = re.match(r"^Europe ([a-z_]+):\s*(\d+)", msg)
            if m:
                mkt, cnt_str = m.group(1), m.group(2)
                cnt = int(cnt_str)
                quota = ref.EU_QUOTAS.get(mkt)
                if quota is not None and cnt < quota:
                    universe_source_warnings.append(
                        f"Europe {mkt} returned {cnt} assets, below quota of {quota}"
                    )

    (outdir / "reference_run.log").write_text("\n".join(ref_log_messages) + "\n", encoding="utf-8")

    # Compute SHA-256 hashes of generated output files
    csv_files = [
        "metrics.csv", "segment_metrics.csv", "daily_returns.csv", "selections.csv",
        "selection_market_counts.csv", "universe_coverage.csv",
        "forward_return_diagnostics.csv", "forward_return_summary.csv",
        "displacement_diagnostics.csv", "policy_delta_diagnostics.csv",
        "policy_delta_summary.json", "tsm_selection_diagnostics.csv",
        "portfolio_comparison.csv", "reference_run.log",
    ]
    file_hashes = {name: compute_file_hash(outdir / name) for name in csv_files if (outdir / name).exists()}

    ref_path = Path("/tmp/global_r63_reference.py")
    ref_sha256 = compute_file_hash(ref_path) if ref_path.exists() else ""
    acquisition_utc = datetime.now(timezone.utc).isoformat()

    metadata = {
        "review_only": True,
        "not_approved_for_live_trading": True,
        "live_order_enabled": False,
        "status": "REFERENCE_ONLY",
        "payload_verified": True,
        "reference_script_sha256": ref_sha256,
        "expected_compressed_payload_sha256": EXPECTED_COMPRESSED_SHA256,
        "expected_decoded_script_sha256": EXPECTED_DECODED_SHA256,
        "acquisition_timestamp_utc": acquisition_utc,
        "raw_baseline_parity": raw_baseline_parity,
        "universe_source_warnings": universe_source_warnings,
        "constants": {
            "START": ref.START,
            "TRADE_START": ref.TRADE_START,
            "END": ref.END,
            "TOP_N": ref.TOP_N,
            "REBALANCE_STEP": ref.REBALANCE_STEP,
            "COST_RATE": ref.COST_RATE,
            "MIN_POINTS": ref.MIN_POINTS,
        },
        "period": {"start": ref.TRADE_START, "end": "2026-07-07"},
        "strategy": {
            "signal": "r63 JPY total-price proxy & within-sleeve percentile rank",
            "top_n": ref.TOP_N,
            "rebalance_us_sessions": ref.REBALANCE_STEP,
            "transaction_cost_one_way_bps": ref.COST_RATE * 10000,
            "defense": "stock native-r63 median<0 and best approved bond native-r63>8%; 100% best bond",
            "normalization": "per-date within-sleeve rank(pct=True), all EU_QUOTAS markets mapped to europe sleeve",
            "sorting_key": "(-score, -raw, ticker)",
        },
        "candidate_policy": {
            "scenario": "no_canada_tsm_usa_sleeve_market_percentile",
            "canada_excluded": True,
            "scanner_taiwan_excluded": True,
            "tsm_included": True,
            "tsm_market": "taiwan_us",
            "tsm_ranking_sleeve": "usa",
            "ranking_mode": "market_percentile",
        },
        "universe_raw_counts": universe_counts_raw,
        "universe_available_counts": {k: len(v) for k, v in groups_available.items()},
        "downloaded_tickers": int(native.shape[1]),
        "quality_exclusions": exclusions,
        "displacement_summary": disp_summary,
        "policy_delta_summary": policy_delta_summary,
        "output_hashes": file_hashes,
        "caveats": [
            "Current constituents/current broker-access proxies are retrospectively applied; not point-in-time membership.",
            "Equal-weight (EW) benchmark universes differ across scenarios; candidate scenario EW benchmark reflects the candidate universe (excluding Canada and scanner Taiwan, including TSM).",
            "Korea is current SBI list intersected with TradingView Korea large-cap ordering, not an official KRX300 PIT history.",
            "Europe and Hong Kong are current large/mid-cap proxies on Saxo-supported exchanges.",
            "Canada and Taiwan use current US-listed foreign issuers as Webull/Saxo-accessible proxies.",
            "Yahoo adjusted close and public FX series are used; execution, taxes, spreads, and broker minimum fees are not fully modeled.",
            "Signal uses close data and ranking information available at the rebalance close; holdings earn returns from the next session.",
            "Using rank(pct=True) within each sleeve equalizes weight/percentile scale across sleeves regardless of sleeve size, which can overweight small sleeves such as Taiwan relative to their total market capitalization or universe count.",
            "Forward and displacement returns are gross, unweighted arithmetic diagnostics and omit transaction costs, so realized metrics govern.",
            "REFERENCE_ONLY analysis; not verified or approved for live capital allocation.",
        ],
    }

    sanitized_metadata = sanitize_for_json(metadata)
    (outdir / "run_metadata.json").write_text(
        json.dumps(sanitized_metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    report = [
        "# Global r63 market normalization reference backtest",
        "",
        "- review_only: true",
        "- not_approved_for_live_trading: true",
        "- live_order_enabled: false",
        "- status: REFERENCE_ONLY",
        "",
        "## Performance Metrics",
        "",
        metrics_df.to_markdown(floatfmt=".4f"),
        "",
        "## Segment Metrics (Chronological)",
        "",
        segment_metrics_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Forward Return Summary by Scenario & Sleeve",
        "",
        fwd_summary_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Market & Sleeve Selection Slots",
        "",
        selection_counts.to_markdown(index=False) if not selection_counts.empty else "No selections.",
        "",
        "## Displacement & Forward-Return Diagnostics Summary",
        "",
        pd.DataFrame([disp_summary]).to_markdown(index=False),
        "",
        "## Policy Delta Summary",
        "",
        pd.DataFrame([policy_delta_summary["average_metrics_non_regime"]]).to_markdown(index=False),
        "",
        "## Baseline / Candidate Portfolio Comparison",
        "",
        portfolio_comparison_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## TSM Selection Diagnostics",
        "",
        tsm_selection_df.to_markdown(index=False) if not tsm_selection_df.empty else "TSM was not selected in any rebalance window.",
        "",
        "## File Hashes (SHA-256)",
        "",
        pd.DataFrame([{"file": k, "sha256": v} for k, v in file_hashes.items()]).to_markdown(index=False),
        "",
        "## Caveats",
        "",
    ]
    report.extend(f"- {item}" for item in metadata["caveats"])
    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")

    ref.log(metrics_df.to_string())
    ref.log(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
