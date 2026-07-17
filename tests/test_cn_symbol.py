"""Unit tests for China A/B/Beijing-share symbol helpers (symbol_utils)."""

import pytest

from tradingagents.dataflows.symbol_utils import (
    cn_code6,
    cn_sina_symbol,
    cn_xueqiu_symbol,
    is_cn_share,
)


@pytest.mark.unit
class TestCnCode6:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("002594", "002594"),
            ("002594.SZ", "002594"),
            ("SZ002594", "002594"),
            ("sz002594", "002594"),
            ("600519.SS", "600519"),
            ("sh600519", "600519"),
            ("688981", "688981"),   # STAR board
            ("300750", "300750"),   # ChiNext
            ("430047", "430047"),   # Beijing
        ],
    )
    def test_extracts_bare_code(self, raw, expected):
        assert cn_code6(raw) == expected

    @pytest.mark.parametrize("raw", ["AAPL", "TSM", "BTC-USD", "", "002594.XX", "12345", "1234567", None])
    def test_non_cn_returns_none(self, raw):
        assert cn_code6(raw) is None


@pytest.mark.unit
class TestIsCnShare:
    @pytest.mark.parametrize("raw", ["002594", "002594.SZ", "600519.SS", "SZ002594"])
    def test_cn_symbols_true(self, raw):
        assert is_cn_share(raw) is True

    @pytest.mark.parametrize("raw", ["AAPL", "BTC-USD", "EURUSD", "GC=F"])
    def test_non_cn_false(self, raw):
        assert is_cn_share(raw) is False


@pytest.mark.unit
class TestPerVendorForms:
    def test_sina_lowercase_prefix(self):
        assert cn_sina_symbol("002594.SZ") == "sz002594"
        assert cn_sina_symbol("600519.SS") == "sh600519"
        assert cn_sina_symbol("SZ002594") == "sz002594"

    def test_xueqiu_uppercase_prefix(self):
        assert cn_xueqiu_symbol("002594.SZ") == "SZ002594"
        assert cn_xueqiu_symbol("600519") == "SH600519"

    def test_exchange_inferred_from_leading_digit(self):
        # No suffix — exchange inferred from the code's leading digit(s).
        assert cn_sina_symbol("300750") == "sz300750"   # ChiNext -> SZ
        assert cn_sina_symbol("688981") == "sh688981"   # STAR -> SH
        assert cn_xueqiu_symbol("002594") == "SZ002594"

    @pytest.mark.parametrize("fn", [cn_sina_symbol, cn_xueqiu_symbol])
    def test_non_cn_returns_none(self, fn):
        assert fn("AAPL") is None
