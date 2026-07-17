"""LangChain tool wrapper for the deterministic fair-value calculator."""
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.fair_value import calculate_fair_value


@tool
def get_fair_value(
    ticker: Annotated[str, "Ticker symbol, e.g. '002594.SZ' or '002594'"],
) -> str:
    """Calculate fair value using 5 deterministic academic models.

    Implements Gordon Growth, Graham Formula, Residual Income (Ohlson),
    PEG (PEG=1.0), and PB Normalization. All inputs are source-computed
    financial data from akshare — no analyst estimates, no LLM judgment.
    Same inputs always produce the same output.

    Returns a markdown report with per-model fair value + a range
    (low / median / high). Use this as the valuation anchor — do NOT
    invent your own fair value number.

    Args:
        ticker: CN stock symbol.

    Returns:
        Markdown string: model results table + fair value range.
    """
    return calculate_fair_value(ticker)
