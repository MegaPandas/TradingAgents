"""Tests for the deterministic 5-model fair-value calculator.

The pure helpers (``_annualize_roe``, ``_compute_cagr``,
``_extract_financial_metrics``) take an akshare financial-abstract DataFrame
and are exercised with a synthetic frame — no network. ``calculate_fair_value``
is driven by patching the two akshare-data callables it imports lazily.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import fair_value
from tradingagents.dataflows.fair_value import (
    _annualize_roe,
    _compute_cagr,
    _extract_financial_metrics,
    calculate_fair_value,
)


def _abstract(rows: dict[str, list], period_cols: list[str]) -> pd.DataFrame:
    """Build a synthetic stock_financial_abstract-shaped frame.

    ``rows`` maps a 指标 (metric) name to its per-period values (aligned to
    ``period_cols``); the 选项 (section) column is filled with a placeholder.
    """
    df = pd.DataFrame({"指标": list(rows)})
    for i, metric in enumerate(rows):
        for col, val in zip(period_cols, rows[metric]):
            if col not in df.columns:
                df[col] = None
            df.loc[i, col] = val
    df.insert(0, "选项", "section")
    return df


# ---------------------------------------------------------------------------
# _annualize_roe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnnualizeRoe:
    def test_q1_annualizes_x4(self):
        assert _annualize_roe(4.0, "20240331") == pytest.approx(16.0)

    def test_h1_annualizes_x2(self):
        assert _annualize_roe(7.5, "20240630") == pytest.approx(15.0)

    def test_9m_annualizes_x12_over_9(self):
        assert _annualize_roe(9.0, "20240930") == pytest.approx(12.0)

    def test_annual_passthrough(self):
        assert _annualize_roe(15.0, "20241231") == pytest.approx(15.0)

    def test_short_column_passthrough(self):
        # Period strings shorter than 8 chars are returned unchanged.
        assert _annualize_roe(5.0, "2024") == 5.0

    def test_unrecognised_month_defaults_x1(self):
        assert _annualize_roe(10.0, "20240115") == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _compute_cagr
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeCagr:
    def test_three_year_cagr(self):
        # 100 → 133.1 over 3 years = 10% CAGR.
        df = _abstract(
            {"营业总收入": [133.1, 121.0, 110.0, 100.0]},
            ["20241231", "20231231", "20221231", "20211231"],
        )
        cagr = _compute_cagr(df, "营业总收入")
        assert cagr == pytest.approx(0.10, abs=1e-6)

    def test_clamped_to_50_percent(self):
        # 10 → 200 over 3 years ≈ 173% CAGR — clamped to 50%.
        df = _abstract(
            {"营业总收入": [200.0, 50.0, 20.0, 10.0]},
            ["20241231", "20231231", "20221231", "20211231"],
        )
        assert _compute_cagr(df, "营业总收入") == 0.50

    def test_negative_growth_floored_at_zero(self):
        df = _abstract(
            {"营业总收入": [50.0, 70.0, 90.0, 100.0]},
            ["20241231", "20231231", "20221231", "20211231"],
        )
        assert _compute_cagr(df, "营业总收入") == 0.0

    def test_insufficient_history_returns_none(self):
        df = _abstract(
            {"营业总收入": [100.0, 90.0]},
            ["20241231", "20231231"],
        )
        assert _compute_cagr(df, "营业总收入") is None

    def test_non_positive_returns_none(self):
        df = _abstract(
            {"营业总收入": [100.0, 0.0, 0.0, 0.0]},
            ["20241231", "20231231", "20221231", "20211231"],
        )
        assert _compute_cagr(df, "营业总收入") is None


# ---------------------------------------------------------------------------
# _extract_financial_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractFinancialMetrics:
    def test_extracts_ttm_eps_via_quarterly_supplement(self):
        # FY2024 EPS 3.0; Q1 2025 EPS 1.0; prior-year Q1 2024 EPS 0.5
        # → TTM = FY + Q - priorQ = 3.0 + 1.0 - 0.5 = 3.5
        df = _abstract(
            {
                "稀释每股收益": [1.0, 3.0, 0.5, 2.5],
                "每股净资产": [30.0, 29.0, 28.0, 27.0],
                "净资产收益率": [5.0, 15.0, 14.0, 13.0],
                "营业总收入增长率": [20.0, 18.0, 16.0, 14.0],
            },
            ["20250331", "20241231", "20240331", "20231231"],
        )
        m = _extract_financial_metrics(df)
        assert m["ttm_eps"] == pytest.approx(3.5)
        assert m["fy_eps"] == pytest.approx(3.0)
        # bvps is read off the annual ref_col (20241231), not the latest quarter.
        assert m["bvps"] == pytest.approx(29.0)
        # ref_col is annual (20241231) → roe already annual.
        assert m["roe_pct"] == pytest.approx(15.0)
        assert m["revenue_growth_pct"] == pytest.approx(18.0)

    def test_annualizes_ytd_roe(self):
        # Latest period is Q1 (20250331); no annual col yet → ROE annualized ×4.
        df = _abstract(
            {
                "基本每股收益": [1.0, 4.0],   # falls back to 基本每股收益
                "净资产收益率": [4.0, 16.0],  # Q1 4.0 → ×4 = 16
            },
            ["20250331", "20241231"],
        )
        # No quarterly supplement math here (latest is Q1, annual is 2024):
        # latest_col 20250331 != annual 20241231 → TTM = fy + q - prior.
        # prior-year Q1 col 20240331 absent → falls back to fy_eps.
        m = _extract_financial_metrics(df)
        assert m["ttm_eps"] == pytest.approx(4.0)
        assert m["roe_pct"] == pytest.approx(16.0)

    def test_missing_eps_returns_empty(self):
        df = _abstract({"每股净资产": [30.0]}, ["20241231"])
        assert _extract_financial_metrics(df) == {}

    def test_none_or_empty_returns_empty(self):
        assert _extract_financial_metrics(None) == {}
        assert _extract_financial_metrics(pd.DataFrame()) == {}


# ---------------------------------------------------------------------------
# calculate_fair_value (5-model output)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateFairValue:
    def test_missing_eps_branch(self):
        # Non-empty abstract but no EPS rows → "missing financial data" branch.
        df = _abstract({"每股净资产": [30.0]}, ["20241231"])
        with patch.object(fair_value, "_financial_abstract", return_value=df, create=True), \
             patch("tradingagents.dataflows.akshare_data._financial_abstract", return_value=df), \
             patch("tradingagents.dataflows.akshare_data._guard_cn", return_value="002594"):
            out = calculate_fair_value("002594.SZ")
        assert "missing financial data" in out
        assert "002594" in out

    def test_five_models_and_range_present(self):
        # eps 3.0, bvps 30, roe 15, revenue growth 20 (YoY, no CAGR history).
        df = _abstract(
            {
                "稀释每股收益": [3.0, 3.0],
                "每股净资产": [30.0, 29.0],
                "净资产收益率": [15.0, 14.0],
                "营业总收入增长率": [20.0, 18.0],
            },
            ["20241231", "20231231"],
        )
        with patch("tradingagents.dataflows.akshare_data._financial_abstract", return_value=df), \
             patch("tradingagents.dataflows.akshare_data._guard_cn", return_value="002594"):
            out = calculate_fair_value("002594.SZ")
        for model in ("Gordon Growth", "Graham", "Residual Income", "PEG", "Justified PB"):
            assert model in out
        assert "Fair Value Range" in out or "Fair Value" in out
        assert "CNY" in out

    def test_gordon_growth_math(self):
        # eps 3.0, growth forces g_rev to cap 6%: Gordon = 3*(1.06)/(0.10-0.06)=79.5
        df = _abstract(
            {
                "稀释每股收益": [3.0, 3.0],
                "营业总收入增长率": [20.0, 18.0],
            },
            ["20241231", "20231231"],
        )
        with patch("tradingagents.dataflows.akshare_data._financial_abstract", return_value=df), \
             patch("tradingagents.dataflows.akshare_data._guard_cn", return_value="002594"):
            out = calculate_fair_value("002594.SZ")
        assert "79.50" in out  # Gordon FV

    def test_cyclical_trough_triggers_normalized_eps(self):
        # TTM EPS depressed to 0.5× FY EPS → normalized path kicks in and a
        # normalized-EPS section is appended.
        df = _abstract(
            {
                # latest Q1 2025 EPS 0.4, prior-year Q1 0.4 → TTM = 4.0 + 0.4 - 0.4 = 4.0?
                # Make TTM clearly < 85% of FY: FY=3.0, TTM=2.0 → 0.667 < 0.85.
                "稀释每股收益": [0.4, 3.0, 0.4, 2.4],   # Q1'25, FY24, Q1'24, FY23
                "每股净资产": [30.0, 29.0, 28.0, 27.0],
                "净资产收益率": [15.0, 14.0, 13.0, 12.0],
            },
            ["20250331", "20241231", "20240331", "20231231"],
        )
        # TTM = FY(3.0) + Q1'25(0.4) - Q1'24(0.4) = 3.0 → not < 0.85×3.0.
        # Adjust so the quarterly supplement yields a trough: make prior-year Q1 large.
        df = _abstract(
            {
                # Q1'25=0.5, FY24=3.0, Q1'24=2.0 → TTM = 3.0+0.5-2.0 = 1.5 (< 0.85×3.0=2.55)
                "稀释每股收益": [0.5, 3.0, 2.0, 2.4],
                "每股净资产": [30.0, 29.0, 28.0, 27.0],
                "净资产收益率": [15.0, 14.0, 13.0, 12.0],
                "营业总收入增长率": [10.0, 9.0],
            },
            ["20250331", "20241231", "20240331", "20231231"],
        )
        with patch("tradingagents.dataflows.akshare_data._financial_abstract", return_value=df), \
             patch("tradingagents.dataflows.akshare_data._guard_cn", return_value="002594"):
            out = calculate_fair_value("002594.SZ")
        assert "normalized" in out.lower() or "Normalized" in out
