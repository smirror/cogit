from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from research.global_r63_market_normalization import (
    get_broad_sleeve,
    compute_within_sleeve_percentiles,
    compute_market_percentiles,
    rank_assets_df,
    compute_diagnostics,
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


if __name__ == "__main__":
    unittest.main()
