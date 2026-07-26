from __future__ import annotations

import json
import unittest
import numpy as np
import pandas as pd

from research.global_r63_market_normalization import (
    get_broad_sleeve,
    build_no_canada_tsm_policy,
    compute_within_sleeve_percentiles,
    compute_market_percentiles,
    rank_assets_df,
    compute_diagnostics,
    compute_policy_delta_diagnostics,
    sanitize_for_json,
)


class TestGlobalR63MarketNormalization(unittest.TestCase):

    def test_json_safety_flags_remain_booleans(self):
        sanitized = sanitize_for_json({
            "review_only": True,
            "live_order_enabled": np.bool_(False),
            "missing_metric": np.nan,
        })
        self.assertIs(sanitized["review_only"], True)
        self.assertIs(sanitized["live_order_enabled"], False)
        self.assertIsNone(sanitized["missing_metric"])

    def test_sleeve_mapping(self):
        self.assertEqual(get_broad_sleeve("usa"), "usa")
        self.assertEqual(get_broad_sleeve("america"), "usa")
        self.assertEqual(get_broad_sleeve("us"), "usa")
        self.assertEqual(get_broad_sleeve("japan"), "japan")
        self.assertEqual(get_broad_sleeve("korea"), "korea")
        self.assertEqual(get_broad_sleeve("uk"), "europe")
        self.assertEqual(get_broad_sleeve("germany"), "europe")
        self.assertEqual(get_broad_sleeve("france"), "europe")
        self.assertEqual(get_broad_sleeve("switzerland"), "europe")
        self.assertEqual(get_broad_sleeve("netherlands"), "europe")
        self.assertEqual(get_broad_sleeve("sweden"), "europe")
        self.assertEqual(get_broad_sleeve("denmark"), "europe")
        self.assertEqual(get_broad_sleeve("norway"), "europe")
        self.assertEqual(get_broad_sleeve("finland"), "europe")
        self.assertEqual(get_broad_sleeve("italy"), "europe")
        self.assertEqual(get_broad_sleeve("spain"), "europe")
        self.assertEqual(get_broad_sleeve("belgium"), "europe")
        self.assertEqual(get_broad_sleeve("austria"), "europe")
        self.assertEqual(get_broad_sleeve("poland"), "europe")
        self.assertEqual(get_broad_sleeve("portugal"), "europe")
        self.assertEqual(get_broad_sleeve("hongkong"), "hong_kong")
        self.assertEqual(get_broad_sleeve("hong_kong"), "hong_kong")
        self.assertEqual(get_broad_sleeve("canada_us"), "canada")
        self.assertEqual(get_broad_sleeve("canada"), "canada")
        self.assertEqual(get_broad_sleeve("taiwan_us"), "taiwan")
        self.assertEqual(get_broad_sleeve("taiwan"), "taiwan")
        self.assertEqual(get_broad_sleeve("bond"), "bond")
        self.assertEqual(get_broad_sleeve("unknown_market"), "unknown_market")

    def test_percentile_calculation(self):
        r63_row = pd.Series({
            "AAPL": 0.40,
            "MSFT": 0.20,
            "7203.T": 0.10,
            "6758.T": 0.30,
        })
        ticker_to_sleeve = {
            "AAPL": "usa",
            "MSFT": "usa",
            "7203.T": "japan",
            "6758.T": "japan",
        }
        res = rank_assets_df(r63_row, ticker_to_sleeve, mode="market_percentile")
        pct_map = dict(zip(res["ticker"], res["ranking_score"]))
        self.assertAlmostEqual(pct_map["AAPL"], 1.0)
        self.assertAlmostEqual(pct_map["MSFT"], 0.5)
        self.assertAlmostEqual(pct_map["6758.T"], 1.0)
        self.assertAlmostEqual(pct_map["7203.T"], 0.5)

        # Test alias compute_market_percentiles
        pcts = compute_market_percentiles(r63_row, ticker_to_sleeve)
        self.assertAlmostEqual(pcts["AAPL"], 1.0)
        self.assertAlmostEqual(pcts["MSFT"], 0.5)

    def test_deterministic_tie_break(self):
        # Tie break order in market_percentile: ranking_score desc, r63_jpy desc, ticker asc
        # Tickers with same score and same raw r63 should be sorted by ticker asc
        r63_row = pd.Series({
            "MSFT": 0.30,
            "AAPL": 0.30,
            "GOOG": 0.30,
        })
        ticker_to_sleeve = {"MSFT": "usa", "AAPL": "usa", "GOOG": "usa"}
        res = rank_assets_df(r63_row, ticker_to_sleeve, mode="market_percentile")
        self.assertEqual(list(res["ticker"]), ["AAPL", "GOOG", "MSFT"])

        # Tickers with same score but different raw r63 should sort by raw r63 desc
        r63_row2 = pd.Series({
            "AAPL": 0.40,   # top of usa -> score 1.0, raw 0.40
            "6758.T": 0.50, # top of japan -> score 1.0, raw 0.50
        })
        ticker_to_sleeve2 = {"AAPL": "usa", "6758.T": "japan"}
        res2 = rank_assets_df(r63_row2, ticker_to_sleeve2, mode="market_percentile")
        self.assertEqual(list(res2["ticker"]), ["6758.T", "AAPL"])

    def test_raw_mode_percentile_none_and_ranking(self):
        r63_row = pd.Series({
            "AAPL": 0.40,
            "MSFT": 0.20,
            "7203.T": 0.10,
        })
        ticker_to_sleeve = {"AAPL": "usa", "MSFT": "usa", "7203.T": "japan"}
        df_raw = rank_assets_df(r63_row, ticker_to_sleeve, mode="raw")

        for _, row in df_raw.iterrows():
            self.assertIsNone(row["percentile"])
            self.assertEqual(row["ranking_score"], row["r63_jpy"])

        self.assertEqual(list(df_raw["ticker"]), ["AAPL", "MSFT", "7203.T"])

    def test_raw_vs_percentile_rankings_differ_synthetic_data(self):
        # Synthetic cross-market data where USA has higher raw r63 than Japan
        r63_row = pd.Series({
            "US_1": 0.50,
            "US_2": 0.40,
            "US_3": 0.30,
            "JP_1": 0.25, # #1 in Japan
            "JP_2": 0.15,
            "JP_3": 0.05,
        })
        ticker_to_sleeve = {
            "US_1": "usa", "US_2": "usa", "US_3": "usa",
            "JP_1": "japan", "JP_2": "japan", "JP_3": "japan",
        }
        top_n = 2

        raw_df = rank_assets_df(r63_row, ticker_to_sleeve, mode="raw")
        raw_top2 = set(raw_df.head(top_n)["ticker"])

        pct_df = rank_assets_df(r63_row, ticker_to_sleeve, mode="market_percentile")
        pct_top2 = set(pct_df.head(top_n)["ticker"])

        self.assertEqual(raw_top2, {"US_1", "US_2"})
        self.assertEqual(pct_top2, {"US_1", "JP_1"})

        self.assertNotEqual(raw_top2, pct_top2)
        self.assertEqual(len(raw_top2), top_n)
        self.assertEqual(len(pct_top2), top_n)

    def test_forward_diagnostic_alignment(self):
        class MockRef:
            REBALANCE_STEP = 19

            def rebalance_dates(self, calendar):
                return [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-02-01")]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", "2021-02-01", freq="B")
        jpy_close = pd.DataFrame({
            "AAPL": np.linspace(100, 110, len(dates)),
            "MSFT": np.linspace(200, 190, len(dates)),
        }, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {
                "scenario": "all5_raw",
                "date": "2021-01-04",
                "ticker": "AAPL",
                "market": "usa",
                "sleeve": "usa",
                "weight": 0.5,
                "regime": False,
                "r63_jpy": 0.1,
                "score": 0.1,
                "percentile": None,
            },
            {
                "scenario": "all5_market_percentile",
                "date": "2021-01-04",
                "ticker": "AAPL",
                "market": "usa",
                "sleeve": "usa",
                "weight": 0.5,
                "regime": False,
                "r63_jpy": 0.1,
                "score": 1.0,
                "percentile": 1.0,
            },
        ])

        fwd_df, summary_df, disp_df, disp_summary = compute_diagnostics(ref, selections, jpy_close, native_close)

        self.assertEqual(len(fwd_df), 2)
        self.assertAlmostEqual(fwd_df.iloc[0]["forward_21d_return"], 0.10)
        self.assertAlmostEqual(fwd_df.iloc[1]["forward_21d_return"], 0.10)

        self.assertEqual(len(summary_df), 2)
        self.assertAlmostEqual(summary_df.loc[summary_df["scenario"] == "all5_raw", "mean"].iloc[0], 0.10)
        self.assertAlmostEqual(summary_df.loc[summary_df["scenario"] == "all5_market_percentile", "mean"].iloc[0], 0.10)

    def test_displacement_counts_and_jaccard(self):
        class MockRef:
            REBALANCE_STEP = 5

            def rebalance_dates(self, calendar):
                return [calendar[0]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=10, freq="B")
        jpy_close = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 120, 10),
            "C": np.linspace(100, 105, 10),
            "D": np.linspace(100, 115, 10),
        }, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {"scenario": "all5_raw", "date": "2021-01-04", "ticker": "A", "market": "usa", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.1, "score": 0.1, "percentile": None},
            {"scenario": "all5_raw", "date": "2021-01-04", "ticker": "B", "market": "usa", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.2, "score": 0.2, "percentile": None},
            {"scenario": "all5_raw", "date": "2021-01-04", "ticker": "C", "market": "usa", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.05, "score": 0.05, "percentile": None},
            {"scenario": "all5_market_percentile", "date": "2021-01-04", "ticker": "B", "market": "usa", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.2, "score": 1.0, "percentile": 1.0},
            {"scenario": "all5_market_percentile", "date": "2021-01-04", "ticker": "C", "market": "usa", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.05, "score": 0.8, "percentile": 0.8},
            {"scenario": "all5_market_percentile", "date": "2021-01-04", "ticker": "D", "market": "korea", "sleeve": "korea", "weight": 0.33, "regime": False, "r63_jpy": 0.15, "score": 1.0, "percentile": 1.0},
        ])

        fwd_df, summary_df, disp_df, disp_summary = compute_diagnostics(ref, selections, jpy_close, native_close)

        self.assertEqual(len(disp_df), 1)
        row = disp_df.iloc[0]
        self.assertEqual(row["raw_count"], 3)
        self.assertEqual(row["pct_count"], 3)
        self.assertEqual(row["overlap_count"], 2)
        self.assertEqual(row["displaced_count"], 1)
        self.assertEqual(row["added_count"], 1)
        self.assertAlmostEqual(row["jaccard_similarity"], 2.0 / 4.0)

    def test_incomplete_window_exclusion(self):
        class MockRef:
            REBALANCE_STEP = 21

            def rebalance_dates(self, calendar):
                return [calendar[0], calendar[21]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=26, freq="B")
        jpy_close = pd.DataFrame({"AAPL": np.linspace(100, 110, 26)}, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {"scenario": "all5_raw", "date": dates[0].date().isoformat(), "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 0.1, "percentile": None},
            {"scenario": "all5_market_percentile", "date": dates[0].date().isoformat(), "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
            {"scenario": "all5_raw", "date": dates[21].date().isoformat(), "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 0.1, "percentile": None},
            {"scenario": "all5_market_percentile", "date": dates[21].date().isoformat(), "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
        ])

        fwd_df, summary_df, disp_df, disp_summary = compute_diagnostics(ref, selections, jpy_close, native_close)

        win1_rows = fwd_df[fwd_df["date"] == dates[0].date().isoformat()]
        win2_rows = fwd_df[fwd_df["date"] == dates[21].date().isoformat()]
        self.assertTrue(all(win1_rows["complete_window"]))
        self.assertEqual(list(win1_rows["sessions_held"]), [21, 21])
        self.assertFalse(any(win2_rows["complete_window"]))
        self.assertEqual(list(win2_rows["sessions_held"]), [4, 4])

        self.assertEqual(summary_df.loc[summary_df["scenario"] == "all5_raw", "count"].iloc[0], 1)
        self.assertEqual(disp_summary["complete_non_regime_rebalance_dates"], 1)
        self.assertEqual(disp_summary["total_rebalance_dates"], 2)

    def test_candidate_universe_exclusions_and_tsm_inclusion(self):
        groups_available = {
            "base": ["SPY", "AAPL"],
            "korea": ["005930.KS"],
            "europe": ["ASML"],
            "hong_kong": ["0700.HK"],
            "canada": ["SHOP", "RY.TO"],
            "taiwan": ["2330.TW"],
        }
        sleeve_by_ticker = {
            "SPY": "usa", "AAPL": "usa", "005930.KS": "korea",
            "ASML": "europe", "0700.HK": "hong_kong", "SHOP": "canada",
            "RY.TO": "canada", "2330.TW": "taiwan", "TSM": "taiwan",
        }
        no_canada_tsm_pool, candidate_sleeve_map = build_no_canada_tsm_policy(
            groups_available, sleeve_by_ticker
        )
        self.assertNotIn("SHOP", no_canada_tsm_pool)
        self.assertNotIn("RY.TO", no_canada_tsm_pool)
        self.assertNotIn("2330.TW", no_canada_tsm_pool)
        self.assertIn("TSM", no_canada_tsm_pool)
        self.assertEqual(no_canada_tsm_pool.count("TSM"), 1)
        self.assertEqual(candidate_sleeve_map["TSM"], "usa")

    def test_build_no_canada_tsm_policy_with_overlapping_ticker(self):
        groups_available = {
            "base": ["SPY", "SHOP"],
            "korea": ["005930.KS"],
            "europe": ["ASML"],
            "hong_kong": ["0700.HK"],
            "canada": ["SHOP", "RY.TO"],
            "taiwan": ["2330.TW"],
        }
        sleeve_by_ticker = {
            "SPY": "usa", "SHOP": "canada", "005930.KS": "korea",
            "ASML": "europe", "0700.HK": "hong_kong", "RY.TO": "canada",
            "2330.TW": "taiwan", "TSM": "taiwan",
        }
        candidate_pool, candidate_sleeve_map = build_no_canada_tsm_policy(
            groups_available, sleeve_by_ticker
        )
        banned = {"SHOP", "RY.TO", "2330.TW"}
        self.assertNotIn("SHOP", candidate_pool)
        self.assertNotIn("RY.TO", candidate_pool)
        self.assertNotIn("2330.TW", candidate_pool)
        self.assertIn("TSM", candidate_pool)
        self.assertEqual(candidate_pool.count("TSM"), 1)
        self.assertEqual(len(set(candidate_pool) & banned), 0)
        self.assertEqual(candidate_sleeve_map["TSM"], "usa")

    def test_tsm_asset_dedupe_order_and_source_symbol(self):
        class MockAsset:
            def __init__(self, ticker, market, currency, source_symbol, name):
                self.ticker = ticker
                self.market = market
                self.currency = currency
                self.source_symbol = source_symbol
                self.name = name

        tsm_asset = MockAsset("TSM", "taiwan_us", "USD", "NYSE:TSM", "TSM ADR")
        other_tsm = MockAsset("TSM", "taiwan", "TWD", "TWSE:2330", "Taiwan Semi")

        def dedupe_assets(assets):
            seen = set()
            out = []
            for a in assets:
                if a.ticker not in seen:
                    seen.add(a.ticker)
                    out.append(a)
            return out

        deduped = dedupe_assets([tsm_asset, other_tsm])
        resolved_tsm = next((a for a in deduped if a.ticker == "TSM"), None)
        self.assertIsNotNone(resolved_tsm)
        self.assertEqual(resolved_tsm.source_symbol, "NYSE:TSM")

    def test_regime_mismatch_exclusion_from_stats(self):
        class MockRef:
            REBALANCE_STEP = 3

            def rebalance_dates(self, calendar):
                return [calendar[0], calendar[3]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=7, freq="B")
        d0_str = dates[0].date().isoformat()
        d3_str = dates[3].date().isoformat()

        jpy_close = pd.DataFrame({
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "B": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0],
            "C": [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0],
            "TSM": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0],
        }, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            # Date 0: Baseline regime True, Candidate regime False (Regime Mismatch)
            {"scenario": "all5_market_percentile", "date": d0_str, "ticker": "A", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": True, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d0_str, "ticker": "C", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.2, "score": 1.0, "percentile": 1.0},
            # Date 3: Both non-regime False
            {"scenario": "all5_market_percentile", "date": d3_str, "ticker": "B", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.05, "score": 0.5, "percentile": 0.5},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d3_str, "ticker": "TSM", "market": "taiwan_us", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.2, "score": 1.0, "percentile": 1.0},
        ])

        fwd_df, _, _, _ = compute_diagnostics(ref, selections, jpy_close, native_close)
        delta_df, delta_summary, _ = compute_policy_delta_diagnostics(ref, selections, fwd_df, jpy_close, native_close)

        self.assertEqual(len(delta_df), 2)
        row0 = delta_df[delta_df["date"] == d0_str].iloc[0]
        self.assertTrue(row0["regime_mismatch"])
        self.assertTrue(row0["baseline_regime"])
        self.assertFalse(row0["candidate_regime"])

        counts = delta_summary["rebalance_counts"]
        self.assertEqual(counts["regime_mismatch_count"], 1)
        self.assertIn(d0_str, counts["regime_mismatch_dates"])

        dropped_stats = delta_summary["dropped_selection_statistics"]
        added_stats = delta_summary["added_selection_statistics"]
        self.assertEqual(dropped_stats["count"], 1)
        self.assertEqual(added_stats["count"], 1)

    def test_taiwan_removed_selection_statistics(self):
        class MockRef:
            REBALANCE_STEP = 3

            def rebalance_dates(self, calendar):
                return [calendar[0]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=5, freq="B")
        d0_str = dates[0].date().isoformat()

        jpy_close = pd.DataFrame({
            "2330.TW": [100.0, 102.0, 105.0, 110.0, 110.0],
            "TSM": [100.0, 105.0, 110.0, 120.0, 120.0],
            "AAPL": [100.0, 100.0, 100.0, 100.0, 100.0],
        }, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {"scenario": "all5_market_percentile", "date": d0_str, "ticker": "2330.TW", "market": "taiwan", "sleeve": "taiwan", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d0_str, "ticker": "TSM", "market": "taiwan_us", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.2, "score": 1.0, "percentile": 1.0},
        ])

        fwd_df, _, _, _ = compute_diagnostics(ref, selections, jpy_close, native_close)
        _, delta_summary, _ = compute_policy_delta_diagnostics(ref, selections, fwd_df, jpy_close, native_close)

        tw_stats = delta_summary["taiwan_removed_selection_statistics"]
        self.assertEqual(tw_stats["total_baseline_taiwan_selections"], 1)
        self.assertEqual(tw_stats["unique_taiwan_tickers_displaced"], ["2330.TW"])
        self.assertAlmostEqual(tw_stats["expectancy"], 0.10)

    def test_policy_delta_summary_strict_json_serialization(self):
        class MockRef:
            REBALANCE_STEP = 3

            def rebalance_dates(self, calendar):
                return [calendar[0]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=5, freq="B")
        d0_str = dates[0].date().isoformat()

        jpy_close = pd.DataFrame({"AAPL": [100.0, 102.0, 105.0, 110.0, 110.0]}, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {"scenario": "all5_market_percentile", "date": d0_str, "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d0_str, "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
        ])

        fwd_df, _, _, _ = compute_diagnostics(ref, selections, jpy_close, native_close)
        _, delta_summary, _ = compute_policy_delta_diagnostics(ref, selections, fwd_df, jpy_close, native_close)

        sanitized = sanitize_for_json(delta_summary)
        json_str = json.dumps(sanitized, allow_nan=False, indent=2)

        self.assertIn('"review_only": true', json_str)
        self.assertIn('"not_approved_for_live_trading": true', json_str)
        self.assertIn('"live_order_enabled": false', json_str)
        self.assertIn('"status": "REFERENCE_ONLY"', json_str)

    def test_tsm_market_reporting_vs_ranking_sleeve(self):
        r63_row = pd.Series({
            "TSM": 0.10,
            "AAPL": 0.30,
            "MSFT": 0.20,
        })
        market_by_ticker = {"TSM": "taiwan_us", "AAPL": "america", "MSFT": "america"}
        candidate_sleeve_by_ticker = {"TSM": "usa", "AAPL": "usa", "MSFT": "usa"}

        res = rank_assets_df(r63_row, candidate_sleeve_by_ticker, mode="market_percentile")
        pct_map = dict(zip(res["ticker"], res["ranking_score"]))

        self.assertAlmostEqual(pct_map["TSM"], 1.0 / 3.0)
        self.assertEqual(get_broad_sleeve("taiwan_us"), "taiwan")
        self.assertEqual(market_by_ticker["TSM"], "taiwan_us")
        self.assertEqual(candidate_sleeve_by_ticker["TSM"], "usa")

    def test_policy_delta_diagnostics_alignment(self):
        class MockRef:
            REBALANCE_STEP = 5

            def rebalance_dates(self, calendar):
                return [calendar[0]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=10, freq="B")
        d_str = dates[0].date().isoformat()

        jpy_close = pd.DataFrame({
            "AAPL": np.linspace(100, 110, 10),
            "SHOP": np.linspace(100, 90, 10),
            "ASML": np.linspace(100, 105, 10),
            "TSM": np.linspace(100, 120, 10),
        }, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {"scenario": "all5_market_percentile", "date": d_str, "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
            {"scenario": "all5_market_percentile", "date": d_str, "ticker": "SHOP", "market": "canada_us", "sleeve": "canada", "weight": 0.33, "regime": False, "r63_jpy": 0.05, "score": 0.5, "percentile": 0.5},
            {"scenario": "all5_market_percentile", "date": d_str, "ticker": "ASML", "market": "netherlands", "sleeve": "europe", "weight": 0.33, "regime": False, "r63_jpy": 0.08, "score": 0.8, "percentile": 0.8},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d_str, "ticker": "AAPL", "market": "usa", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.1, "score": 0.9, "percentile": 0.9},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d_str, "ticker": "TSM", "market": "taiwan_us", "sleeve": "usa", "weight": 0.33, "regime": False, "r63_jpy": 0.2, "score": 1.0, "percentile": 1.0},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d_str, "ticker": "ASML", "market": "netherlands", "sleeve": "europe", "weight": 0.33, "regime": False, "r63_jpy": 0.08, "score": 0.8, "percentile": 0.8},
        ])

        fwd_df, summary_df, disp_df, disp_summary = compute_diagnostics(ref, selections, jpy_close, native_close)
        delta_df, delta_summary, tsm_df = compute_policy_delta_diagnostics(ref, selections, fwd_df, jpy_close, native_close)

        self.assertEqual(len(delta_df), 1)
        row = delta_df.iloc[0]
        self.assertEqual(row["baseline_count"], 3)
        self.assertEqual(row["candidate_count"], 3)
        self.assertEqual(row["overlap_count"], 2)
        self.assertEqual(row["dropped_count"], 1)
        self.assertEqual(row["added_count"], 1)
        self.assertAlmostEqual(row["jaccard_similarity"], 2.0 / 4.0)
        self.assertEqual(row["canada_dropped_count"], 1)
        self.assertTrue(row["tsm_selected"])

        self.assertEqual(len(tsm_df), 1)
        self.assertEqual(tsm_df.iloc[0]["ticker"], "TSM")

        can_stats = delta_summary["canada_removed_selection_statistics"]
        self.assertEqual(can_stats["total_baseline_canada_selections"], 1)
        self.assertEqual(can_stats["unique_canada_tickers_displaced"], ["SHOP"])

        tsm_stats = delta_summary["tsm_selection_statistics"]
        self.assertEqual(tsm_stats["total_selections"], 1)

    def test_mae_mfe_holding_window_calculation(self):
        class MockRef:
            REBALANCE_STEP = 5

            def rebalance_dates(self, calendar):
                return [calendar[0], calendar[5]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=8, freq="B")
        d0_str = dates[0].date().isoformat()
        d5_str = dates[5].date().isoformat()

        # Price trajectory for ticker XYZ: 100 on d0, dips to 90 (MAE -10%), peaks at 110 (MFE +10%), 105 at d5
        jpy_close = pd.DataFrame({
            "XYZ": [100.0, 95.0, 90.0, 105.0, 110.0, 105.0, 102.0, 101.0]
        }, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {"scenario": "all5_market_percentile", "date": d0_str, "ticker": "XYZ", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
            {"scenario": "all5_market_percentile", "date": d5_str, "ticker": "XYZ", "market": "usa", "sleeve": "usa", "weight": 1.0, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
        ])

        fwd_df, _, _, _ = compute_diagnostics(ref, selections, jpy_close, native_close)

        # Complete window row (d0_str)
        row0 = fwd_df[fwd_df["date"] == d0_str].iloc[0]
        self.assertTrue(row0["complete_window"])
        self.assertAlmostEqual(row0["forward_21d_return"], 0.05)
        self.assertAlmostEqual(row0["mae"], -0.10)
        self.assertAlmostEqual(row0["mfe"], 0.10)

        # Incomplete window row (d5_str)
        row1 = fwd_df[fwd_df["date"] == d5_str].iloc[0]
        self.assertFalse(row1["complete_window"])
        self.assertTrue(pd.isna(row1["mae"]))
        self.assertTrue(pd.isna(row1["mfe"]))

    def test_profit_factor_and_extended_stats_calculation(self):
        class MockRef:
            REBALANCE_STEP = 3

            def rebalance_dates(self, calendar):
                return [calendar[0]]

        ref = MockRef()
        dates = pd.date_range("2021-01-04", periods=5, freq="B")
        d_str = dates[0].date().isoformat()

        jpy_close = pd.DataFrame({
            "A": [100.0, 102.0, 105.0, 110.0, 110.0],  # +10% return, min 100 -> MAE 0.0
            "B": [100.0, 98.0, 95.0, 95.0, 95.0],     # -5% return, min 95 -> MAE -5%
            "C": [100.0, 101.0, 103.0, 106.0, 106.0],  # +6% return, min 100 -> MAE 0.0
            "TSM": [100.0, 105.0, 110.0, 120.0, 120.0], # +20% return, min 100 -> MAE 0.0
        }, index=dates)
        native_close = jpy_close.copy()

        selections = pd.DataFrame([
            {"scenario": "all5_market_percentile", "date": d_str, "ticker": "A", "market": "usa", "sleeve": "usa", "weight": 0.5, "regime": False, "r63_jpy": 0.1, "score": 1.0, "percentile": 1.0},
            {"scenario": "all5_market_percentile", "date": d_str, "ticker": "B", "market": "canada_us", "sleeve": "canada", "weight": 0.5, "regime": False, "r63_jpy": 0.05, "score": 0.5, "percentile": 0.5},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d_str, "ticker": "C", "market": "usa", "sleeve": "usa", "weight": 0.5, "regime": False, "r63_jpy": 0.08, "score": 0.8, "percentile": 0.8},
            {"scenario": "no_canada_tsm_usa_sleeve_market_percentile", "date": d_str, "ticker": "TSM", "market": "taiwan_us", "sleeve": "usa", "weight": 0.5, "regime": False, "r63_jpy": 0.2, "score": 1.0, "percentile": 1.0},
        ])

        fwd_df, _, _, _ = compute_diagnostics(ref, selections, jpy_close, native_close)
        _, delta_summary, _ = compute_policy_delta_diagnostics(ref, selections, fwd_df, jpy_close, native_close)

        dropped_stats = delta_summary["dropped_selection_statistics"]
        added_stats = delta_summary["added_selection_statistics"]
        canada_stats = delta_summary["canada_removed_selection_statistics"]
        tsm_stats = delta_summary["tsm_selection_statistics"]

        # Dropped tickers are A (+10%, MAE 0.0) and B (-5%, MAE -0.05)
        self.assertEqual(dropped_stats["count"], 2)
        self.assertAlmostEqual(dropped_stats["expectancy"], 0.025)
        self.assertAlmostEqual(dropped_stats["profit_factor"], 2.0)
        self.assertEqual(dropped_stats["profit_factor_state"], "finite")
        self.assertAlmostEqual(dropped_stats["mean_mae"], -0.025)

        # Added C (+6%, TSM +20%) -> added stats has C and TSM (+6%, +20%)
        self.assertEqual(added_stats["count"], 2)
        self.assertAlmostEqual(added_stats["expectancy"], 0.13)
        self.assertIsNone(added_stats["profit_factor"])
        self.assertEqual(added_stats["profit_factor_state"], "no_losing_trades")

        # Canada B
        self.assertEqual(canada_stats["complete_window_canada_selections"], 1)
        self.assertAlmostEqual(canada_stats["expectancy"], -0.05)

        # TSM
        self.assertEqual(tsm_stats["complete_window_selections"], 1)
        self.assertAlmostEqual(tsm_stats["expectancy"], 0.20)


if __name__ == "__main__":
    unittest.main()
