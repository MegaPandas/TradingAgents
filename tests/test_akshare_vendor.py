"""Unit tests for the akshare vendor (China A-share prices/indicators/fundamentals).

akshare is mocked so these run without network or the package installed; they
verify the vendor contract (A-share detection, graceful non-CN -> NoMarketDataError
so the router falls through, output shape, currency tag, error mapping).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import akshare_data, interface
from tradingagents.dataflows.errors import NoMarketDataError, VendorNotConfiguredError

# A 3-row qfq OHLCV frame shaped like akshare's stock_zh_a_daily output.
_OHLCV = pd.DataFrame({
    "date": ["2026-07-08", "2026-07-09", "2026-07-10"],
    "open": [87.0, 86.5, 86.84],
    "high": [88.0, 87.77, 90.83],
    "low": [86.0, 85.80, 84.88],
    "close": [87.8, 86.87, 90.00],
    "volume": [3.0e7, 3.77e7, 7.09e7],
})


def _mock_akshare():
    """A stand-in akshare module exposing the methods the vendor calls."""
    return SimpleNamespace(
        stock_zh_a_daily=lambda **kw: _OHLCV.copy(),
        stock_financial_abstract=lambda symbol: pd.DataFrame({
            "选项": ["资产负债表", "资产负债表", "利润表"],
            "指标": ["货币资金", "应收账款", "营业收入"],
            "20260331": [1e9, 2e8, 5e9],
            "20251231": [9e8, 1.8e8, 4.5e9],
        }),
        stock_financial_analysis_indicator=lambda symbol, start_year: pd.DataFrame({
            "日期": ["2026-03-31"],
            "摊薄每股收益(元)": [3.5],
            "净资产收益率(%)": [12.0],
            "销售毛利率(%)": [20.0],
        }),
        stock_news_em=lambda symbol: pd.DataFrame({
            "关键词": [symbol, symbol],
            "新闻标题": ["比亚迪发布新车", "销量超预期"],
            "新闻内容": ["正文一", "正文二"],
            "发布时间": ["2026-07-09 10:00:00", "2026-07-10 14:30:00"],
            "文章来源": ["财联社", "新浪财经"],
            "新闻链接": ["http://a", "http://b"],
        }),
        macro_china_lpr=lambda: pd.DataFrame({
            "TRADE_DATE": ["2026-06-22"],
            "LPR1Y": [3.0],
            "LPR5Y": [3.5],
        }),
    )


def _mock_akshare_with_daily(df):
    """Like _mock_akshare but with a custom stock_zh_a_daily return value."""
    ak = _mock_akshare()
    ak.stock_zh_a_daily = lambda **kw: df.copy()
    return ak


@pytest.mark.unit
class TestGetStock:
    def test_cn_share_returns_cny_csv(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            out = akshare_data.get_stock("002594.SZ", "2026-07-01", "2026-07-10")
        assert "002594" in out
        assert "Currency: CNY" in out
        assert "Date" in out and "Close" in out  # CSV header
        assert "86.87" in out  # a close value

    def test_non_cn_raises_no_market_data(self):
        # Must raise (not return empty) so the router falls through to yfinance.
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            with pytest.raises(NoMarketDataError):
                akshare_data.get_stock("AAPL", "2026-07-01", "2026-07-10")

    def test_empty_result_raises_no_market_data(self):
        ak = _mock_akshare()
        ak.stock_zh_a_daily = lambda **kw: pd.DataFrame()
        with patch.object(akshare_data, "_require_akshare", return_value=ak):
            with pytest.raises(NoMarketDataError):
                akshare_data.get_stock("002594.SZ", "2026-07-01", "2026-07-10")

    def test_missing_akshare_raises_not_configured(self):
        def _raise():
            raise VendorNotConfiguredError("akshare not installed")
        with patch.object(akshare_data, "_require_akshare", side_effect=_raise):
            with pytest.raises(VendorNotConfiguredError):
                akshare_data.get_stock("002594.SZ", "2026-07-01", "2026-07-10")


@pytest.mark.unit
class TestIndicator:
    def test_cn_indicator_returns_window(self):
        # stockstats needs >= a few rows; the 3-row fixture is enough for a
        # simple indicator like close_5_sma won't compute on 3 rows, so use
        # a longer synthetic frame.
        long_df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d"),
            "open": range(60), "high": range(60), "low": range(60),
            "close": [10 + i * 0.1 for i in range(60)], "volume": [1000] * 60,
        })
        ak = _mock_akshare()
        ak.stock_zh_a_daily = lambda **kw: long_df.copy()
        with patch.object(akshare_data, "_require_akshare", return_value=ak):
            out = akshare_data.get_stock_stats_indicators_window(
                "600519.SS", "close_5_sma", "2025-03-01", look_back_days=5
            )
        assert "Currency: CNY" in out
        assert "close_5_sma" in out

    def test_non_cn_raises(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            with pytest.raises(NoMarketDataError):
                akshare_data.get_stock_stats_indicators_window(
                    "TSM", "rsi_14", "2026-07-10", look_back_days=5
                )


@pytest.mark.unit
class TestFundamentals:
    def test_fundamentals_returns_metrics_and_currency(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            out = akshare_data.get_fundamentals("002594.SZ")
        assert "Currency: CNY" in out
        assert "3.5" in out  # diluted EPS

    def test_balance_sheet_section_filter(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            out = akshare_data.get_balance_sheet("002594.SZ")
        assert "货币资金" in out          # a balance-sheet line kept
        assert "营业收入" not in out       # income line filtered out

    def test_income_statement_section_filter(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            out = akshare_data.get_income_statement("002594.SZ")
        assert "营业收入" in out


@pytest.mark.unit
class TestRoutingFallback:
    def test_akshare_no_data_falls_through_to_yfinance(self):
        """A non-CN symbol: akshare raises NoMarketDataError, router tries yfinance."""
        yf_result = "YFINANCE_OK"

        def fake_yfinance(symbol, start, end):
            return yf_result

        with patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"akshare": akshare_data.get_stock, "yfinance": fake_yfinance}},
            clear=False,
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "AAPL", "2026-07-01", "2026-07-10"
            )
        assert result == yf_result


@pytest.mark.unit
class TestNews:
    def test_cn_news_filtered_to_range(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            out = akshare_data.get_news("002594.SZ", "2026-07-01", "2026-07-10")
        assert "比亚迪发布新车" in out
        assert "akshare/eastmoney" in out

    def test_non_cn_raises(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            with pytest.raises(NoMarketDataError):
                akshare_data.get_news("AAPL", "2026-07-01", "2026-07-10")

    def test_no_news_in_range_raises(self):
        # The mock news are dated 2026-07-09/10; ask for an out-of-range window.
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            with pytest.raises(NoMarketDataError):
                akshare_data.get_news("002594.SZ", "2025-01-01", "2025-01-10")


@pytest.mark.unit
class TestMacro:
    def test_cn_alias_returns(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            out = akshare_data.get_macro_data("lpr", "2026-07-10")
        assert "China LPR" in out
        assert "LPR1Y" in out  # column header preserved

    def test_generic_alias_raises_so_fred_serves(self):
        with patch.object(akshare_data, "_require_akshare", return_value=_mock_akshare()):
            with pytest.raises(NoMarketDataError):
                # 'cpi' is generic (FRED); akshare must not claim it.
                akshare_data.get_macro_data("cpi", "2026-07-10")

    def test_descending_series_returns_newest_rows(self):
        # money_supply / cpi arrive newest-first (descending). The formatter must
        # take the most recent rows by date — NOT df.tail(), which would hand the
        # LLM 2008-era data (the M2/CPI stale-data bug).
        ak = _mock_akshare()
        ak.macro_china_money_supply = lambda: pd.DataFrame({
            "月份": ["2026年05月份", "2025年10月份", "2008年02月份"],
            "货币和准货币(M2)-同比增长": [8.6, 8.2, 17.39],
        })
        with patch.object(akshare_data, "_require_akshare", return_value=ak):
            out = akshare_data.get_macro_data("m2", "2026-07-11")
        assert "2026年05月份" in out
        # newest row must precede the stale 2008 row in the rendered output
        assert out.index("2026年05月份") < out.index("2008年02月份")

    def test_macro_routing_falls_through_to_fred(self):
        fred_result = "FRED_OK"

        def fake_fred(indicator, curr_date, look_back_days=None):
            return fred_result

        with patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_indicators": {
                "akshare": akshare_data.get_macro_data, "fred": fake_fred,
            }},
            clear=False,
        ):
            result = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-07-10", 365)
        assert result == fred_result
