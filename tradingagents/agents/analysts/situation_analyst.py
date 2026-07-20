"""Situation Analyst — explains how the instrument got to its current price.

Reads the other analysts' reports and uses web_search to retrieve
historical catalysts (1-90 days ago) that the News analyst's 7-day
window does not cover. Synthesises a dated narrative: what events moved
the price, how market expectations shifted, and what the current setup
implies.
"""
from __future__ import annotations

from tradingagents.agents.utils.agent_utils import (
    _PIPELINE_PREAMBLE,
    chat_prompt_messages,
    safe_llm_invoke,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
    web_search,
)
from tradingagents.agents.utils.agent_utils import format_price_context


_SYSTEM_TEMPLATE = _PIPELINE_PREAMBLE + """ROLE
You are the Situation Analyst. Explain how the instrument arrived at its current price — the catalyst timeline, the narrative shifts, and the expectations the market has embedded over the past quarter.

SCOPE
1. Price timeline — key price levels (highs, lows, turning points), each dated, each linked to a specific event or data point.
2. Catalyst calendar — earnings reports, product launches, policy changes, macro data, competitor events, in chronological order.
3. Narrative shifts — what the market was pricing BEFORE each catalyst vs AFTER. State the dominant narrative at each stage.
4. Current setup — what expectations are currently embedded in the price. What would need to happen to break the current range in either direction.
5. Fact vs narrative — label which price moves are attributable to specific data vs broader sentiment.

HARD CONSTRAINTS
- Every dated event must cite its source (analyst report, get_news result, or web_search result).
- Do NOT restate the other analysts' conclusions — synthesise the TIMELINE that led to those conclusions.
- Ground every claim in a fetched source or an analyst report.
- Use web_search(query) to retrieve HISTORICAL catalysts (1-90 days ago) that the 7-day news window does not cover: Q1 earnings miss, policy change, product launch, competitor move, analyst downgrade. Ask "what changed and when."
- **SHARE-CLASS DISCIPLINE (Fix 9)**: this node analyzes the A-share listing (e.g. 002594.SZ). If any source report cites a target price or metric from the H-share listing (codes starting with 1, 5, 7 followed by digits, or exchange tag .HK), label it explicitly as the H-share figure: e.g. "CLSA H-share (1211.HK) target 120 HKD — NOT directly comparable to A-share (002594.SZ)". Do NOT translate H-share targets into A-share targets without explicit share-class conversion and exchange-rate note. H-share P/E reflects H-share liquidity/regulatory regime and is generally lower than A-share; do NOT use H-share multiples for A-share valuation.
- The analyst reports are in the DATA block below — read them for timeline events, not conclusions.

OUTPUT (write in this structure)
## 1. Price Journey (past quarter)
## 2. Catalyst Calendar (chronological)
## 3. Narrative Shifts
## 4. Current Setup
## 5. Summary Timeline Table"""


def create_situation_analyst(llm):
    """Create the Situation analyst node — runs after other analysts, before digest."""

    def situation_node(state) -> dict:
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        price_ctx = format_price_context(state)
        price_lines = price_ctx if price_ctx else "PRICE: UNAVAILABLE"

        # Reports stay in data_block, not in the system message — cache hit.
        market_report = state.get("market_report") or "(no market report)"
        sentiment_report = state.get("sentiment_report") or "(no sentiment report)"
        news_report = state.get("news_report") or "(no news report)"
        fundamentals_report = state.get("fundamentals_report") or "(no fundamentals report)"
        macro_policy_report = state.get("macro_policy_report") or "(no macro/policy report)"
        business_report = state.get("business_report") or "(no business report)"

        data_block = (
            f"\n\n{price_lines}\n"
            f"\n===== ANALYST REPORTS ====="
            f"\n--- MARKET ---\n{market_report}"
            f"\n--- SENTIMENT ---\n{sentiment_report}"
            f"\n--- NEWS ---\n{news_report}"
            f"\n--- FUNDAMENTALS ---\n{fundamentals_report}"
            f"\n--- MACRO & POLICY ---\n{macro_policy_report}"
            f"\n--- BUSINESS ---\n{business_report}"
            f"\n===== END REPORTS ====="
        )

        system_message = _SYSTEM_TEMPLATE + get_language_instruction()

        tools = [web_search, get_news]

        prompt = chat_prompt_messages(
            system_message, tools=tools,
            current_date=current_date, instrument_context=instrument_context,
            data_block=data_block,
        )

        chain = prompt | llm.bind_tools(tools)
        result = safe_llm_invoke(chain, state["messages"], "Situation Analyst")

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "situation_report": report,
        }

    return situation_node
