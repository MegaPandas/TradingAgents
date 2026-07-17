"""Tests that empty vendor results never become fabricated data.

Covers two systematic fixes:
  - load_ohlcv must not cache an empty download (cache poisoning), and must
    raise NoMarketDataError instead of returning an empty frame.
  - route_to_vendor must convert NoMarketDataError into a single explicit
    "NO_DATA_AVAILABLE" sentinel after all vendors are exhausted.
"""

import os
import unittest
from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import interface, stockstats_utils
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


@pytest.mark.unit
class TestLoadOhlcvNoPoison(unittest.TestCase):
    def setUp(self):
        self._tmp = os.path.join(os.path.dirname(__file__), "_tmp_cache")
        os.makedirs(self._tmp, exist_ok=True)
        set_config({"data_cache_dir": self._tmp})

    def tearDown(self):
        for f in os.listdir(self._tmp):
            os.remove(os.path.join(self._tmp, f))
        os.rmdir(self._tmp)

    def test_empty_download_raises_and_does_not_cache(self):
        empty = pd.DataFrame()
        with mock.patch.object(stockstats_utils.yf, "download", return_value=empty), \
                self.assertRaises(NoMarketDataError):
            stockstats_utils.load_ohlcv("FAKE", "2026-01-01")
        # Nothing should have been written to the cache.
        self.assertEqual(os.listdir(self._tmp), [])

        # A second call must re-attempt the fetch (no poisoned cache served).
        with mock.patch.object(stockstats_utils.yf, "download", return_value=empty) as dl2:
            with self.assertRaises(NoMarketDataError):
                stockstats_utils.load_ohlcv("FAKE", "2026-01-01")
            self.assertTrue(dl2.called)


@pytest.mark.unit
class TestRouteToVendorSentinel(unittest.TestCase):
    def test_no_data_from_all_vendors_returns_sentinel(self):
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, "GC=F", "no rows")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_no_data}
        with mock.patch.dict(
            interface.VENDOR_METHODS, {"get_stock_data": patched}, clear=False
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "XAUUSD+", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("XAUUSD+", result)
        self.assertIn("GC=F", result)
        self.assertIn("Do not estimate", result)

    def test_unconfigured_fallback_does_not_mask_no_data(self):
        # When the primary vendor reports no data and the fallback is simply
        # unavailable (e.g. missing API key -> raises), the no-data sentinel
        # must win rather than the fallback's incidental error crashing out.
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, symbol, "no rows")

        def raises_unavailable(symbol, *a, **k):
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_unavailable}
        with mock.patch.dict(
            interface.VENDOR_METHODS, {"get_stock_data": patched}, clear=False
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "FAKE", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)


@pytest.mark.unit
class TestLoadOhlcvMultiSource(unittest.TestCase):
    """load_ohlcv tries multiple sources for CN shares (akshare -> yfinance)."""

    def _df(self):
        return pd.DataFrame({
            "Date": pd.to_datetime(["2026-07-09", "2026-07-10"]),
            "Open": [9.0, 9.1], "High": [9.5, 9.6],
            "Low": [8.9, 9.0], "Close": [9.2, 9.3], "Volume": [1000, 2000],
        })

    def test_cn_falls_back_to_yahoo_when_akshare_fails(self):
        ok = self._df()
        with mock.patch.object(
            stockstats_utils, "_load_ohlcv_akshare",
            side_effect=NoMarketDataError("002185.SZ", "002185", "akshare down"),
        ), mock.patch.object(
            stockstats_utils, "_load_ohlcv_yahoo", return_value=ok,
        ) as yf_loader:
            out = stockstats_utils.load_ohlcv("002185.SZ", "2026-07-10")
        self.assertFalse(out.empty)
        yf_loader.assert_called_once()  # backup actually tried

    def test_cn_does_not_call_yahoo_when_akshare_succeeds(self):
        ok = self._df()
        with mock.patch.object(
            stockstats_utils, "_load_ohlcv_akshare", return_value=ok,
        ), mock.patch.object(
            stockstats_utils, "_load_ohlcv_yahoo", return_value=ok,
        ) as yf_loader:
            stockstats_utils.load_ohlcv("002185.SZ", "2026-07-10")
        yf_loader.assert_not_called()

    def test_all_sources_fail_reraises(self):
        with mock.patch.object(
            stockstats_utils, "_load_ohlcv_akshare",
            side_effect=NoMarketDataError("002185.SZ", "002185", "akshare down"),
        ), mock.patch.object(
            stockstats_utils, "_load_ohlcv_yahoo",
            side_effect=NoMarketDataError("002185.SZ", "002185.SZ", "yahoo no rows"),
        ):
            with self.assertRaises(NoMarketDataError):
                stockstats_utils.load_ohlcv("002185.SZ", "2026-07-10")


if __name__ == "__main__":
    unittest.main()
