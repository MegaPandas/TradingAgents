"""Guard the news analyst prompt against tool-signature drift (#1116).

The prompt used to advertise ``get_news(query, ...)`` while the tool takes a
``ticker``, tricking the LLM into hallucinating free-text query calls.
"""
import inspect
from types import SimpleNamespace

import pytest

import tradingagents.agents.analysts.news_analyst as na
from tradingagents.agents.utils.news_data_tools import get_news


@pytest.mark.unit
def test_get_news_takes_ticker_not_query():
    arg_names = set(get_news.args.keys())
    assert "ticker" in arg_names
    assert "query" not in arg_names


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    src = inspect.getsource(na)
    assert "get_news(ticker, start_date, end_date)" in src
    assert "get_news(query" not in src


class _FakeLLM:
    """Records the tools it was bound to; returns an empty (no tool_calls) result."""
    def __init__(self):
        self.bound_tool_names = None

    def bind_tools(self, tools):
        self.bound_tool_names = [t.name for t in tools]
        return self

    def __call__(self, _messages):
        return SimpleNamespace(tool_calls=[], content="")


def _state(ticker):
    return {
        "messages": [],
        "trade_date": "2026-07-10",
        "company_of_interest": ticker,
        "instrument_context": f"instrument {ticker}",
        "asset_type": "stock",
    }


@pytest.mark.unit
def test_cn_share_drops_prediction_markets_and_keeps_macro():
    llm = _FakeLLM()
    node = na.create_news_analyst(llm)
    node(_state("002594.SZ"))
    assert "get_macro_indicators" in llm.bound_tool_names
    assert "get_prediction_markets" not in llm.bound_tool_names


@pytest.mark.unit
def test_non_cn_keeps_prediction_markets():
    llm = _FakeLLM()
    node = na.create_news_analyst(llm)
    node(_state("AAPL"))
    assert "get_macro_indicators" in llm.bound_tool_names
    assert "get_prediction_markets" in llm.bound_tool_names
