"""Tests for Propagator.create_initial_state CN-vs-global gating.

The sina minute-bar price snapshots are CN-only. For US/crypto/global tickers
the pre-fetch must be skipped entirely (no wasted ``import akshare`` or bogus
sina calls), leaving the four latest_close_* fields None.
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.graph.propagation import Propagator


def _news_tool_mock():
    """The propagation path calls ``get_news.func(...)``, so the mock needs a
    callable ``.func`` attribute."""
    tool = MagicMock()
    tool.func = MagicMock(return_value="")
    return tool


@pytest.mark.unit
def test_non_cn_ticker_skips_sina_price_snapshot():
    """AAPL is not a CN share — the akshare price block must not run."""
    propagator = Propagator()
    with patch("tradingagents.graph.propagation.get_news", _news_tool_mock()), \
         patch("tradingagents.dataflows.fair_value.calculate_fair_value", return_value=""), \
         patch("tradingagents.dataflows.akshare_data._no_proxy") as no_proxy_mock:
        state = propagator.create_initial_state(
            "AAPL", "2026-01-15", "stock", past_context="", instrument_context="",
        )
    # The four price fields stay None — no sina fetch attempted.
    assert state["latest_close"] is None
    assert state["latest_close_realtime"] is None
    assert state["latest_close_settled_at"] is None
    assert state["latest_close_realtime_at"] is None
    # _no_proxy is the sina-call context manager — must not be entered for non-CN.
    no_proxy_mock.assert_not_called()


@pytest.mark.unit
def test_non_cn_ticker_still_seeds_required_fields():
    """Even with the price block skipped, the rest of the initial state is sane."""
    propagator = Propagator()
    with patch("tradingagents.graph.propagation.get_news", _news_tool_mock()), \
         patch("tradingagents.dataflows.fair_value.calculate_fair_value", return_value="FV"):
        state = propagator.create_initial_state(
            "BTC-USD", "2026-01-15", "crypto", past_context="", instrument_context="",
        )
    assert state["company_of_interest"] == "BTC-USD"
    assert state["asset_type"] == "crypto"
    assert state["trade_date"] == "2026-01-15"
    # Fair-value pre-fetch still runs (it no-ops internally for non-CN, returns "").
    assert state["fair_value_block"] == "FV"
    # Debate transcripts initialised empty.
    assert state["research_debate_turns"] == []
    assert state["risk_debate_turns"] == []
    assert state["risk_synthesis"] == ""
