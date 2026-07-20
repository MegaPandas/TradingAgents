"""Business Analyst — segment-level revenue/profit/structure breakdown.

Independent of Fundamentals: reads the other analysts' reports for
financial data already fetched. Uses web_search for product pipeline,
new model launches, auto show feedback, capacity expansion plans, and
competitor moves — these are not in financial statements.
"""
from __future__ import annotations

from tradingagents.agents.utils.agent_utils import (
    _PIPELINE_PREAMBLE,
    chat_prompt_messages,
    get_instrument_context_from_state,
    get_language_instruction,
    safe_llm_invoke,
    web_search,
)


_SEGMENT_STRUCTURE = """\
## 1. Revenue Map — reported segments, share %, YoY growth, profit contribution. Flag >20% segments with direction change.
## 1b. Product/Brand Map — sub-brand/model-level sales, ASP, pipeline launches (web_search for auto-show/MIIT data).
## 2. Competitive Position — per-segment market share, top-3 competitors, pricing power, concentration risk.
## 3. Margin by Segment — margin per segment, export vs domestic split, which drives profit vs drag.
## 4. Growth Drivers — volume/price/geo/product drivers, capacity, capex, order backlog.
## 5. Segment Risk Map — per-segment key risk + probability, cannibalisation risk.
## 6. Segment Summary Table — segment name | share % | YoY growth | margin contribution | market position | key risk. Second table for sub-brands: brand | segment | volume | YoY | revenue contribution (est.)"""


_SYSTEM_TEMPLATE = _PIPELINE_PREAMBLE + (
    "ROLE\n"
    "You are the Business Analyst. Break down the company's revenue into its constituent segments and build a competitive portrait per segment. You read the other analysts' reports for financial data already fetched. Use web_search(query) for product pipeline, new model launches, auto show feedback, capacity expansion plans, and competitor moves — these are not in financial statements.\n\n"
    "SCOPE\n"
    "1. Revenue map — every reportable segment, its revenue contribution, growth rate, profit contribution.\n"
    "2. Competitive position — market share, top competitors, pricing power, channel dependency per segment.\n"
    "3. Margin by segment — which segments are profitable, which are margin drag.\n"
    "4. Growth drivers — volume/price/geographic/product-line drivers per segment; capacity and capex.\n"
    "5. Risk — per-segment risk with estimated likelihood.\n\n"
    "HARD CONSTRAINTS\n"
    "- Every revenue/market-share/margin figure MUST cite a source (analyst report or tool output). No unsourced claims.\n"
    "- Do NOT value the company or recommend a rating — that is the Valuation Analyst's job.\n"
    "- Do NOT synthesise a narrative timeline — that is the Situation Analyst's job.\n"
    "- Separate published data from estimate. Label estimates with \"(est.)\".\n"
    "- The analyst reports are in the DATA block below — use them for already-fetched financials and segment breakdown hints.\n\n"
    "OUTPUT (use this exact structure, filling every section)\n"
    + _SEGMENT_STRUCTURE
)


def create_business_analyst(llm):
    """Create the Business analyst node — runs after core analysts, before digest."""

    def business_node(state) -> dict:
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        market_report = state.get("market_report") or "(no market report)"
        sentiment_report = state.get("sentiment_report") or "(no sentiment report)"
        news_report = state.get("news_report") or "(no news report)"
        fundamentals_report = state.get("fundamentals_report") or "(no fundamentals report)"
        macro_policy_report = state.get("macro_policy_report") or "(no macro/policy report)"

        data_block = (
            f"\n\n===== ANALYST REPORTS ====="
            f"\n--- MARKET ---\n{market_report}"
            f"\n--- SENTIMENT ---\n{sentiment_report}"
            f"\n--- NEWS ---\n{news_report}"
            f"\n--- FUNDAMENTALS ---\n{fundamentals_report}"
            f"\n--- MACRO & POLICY ---\n{macro_policy_report}"
            f"\n===== END REPORTS ====="
        )

        system_message = _SYSTEM_TEMPLATE + get_language_instruction()

        tools = [web_search]

        prompt = chat_prompt_messages(
            system_message, tools=tools,
            current_date=current_date, instrument_context=instrument_context,
            data_block=data_block,
        )

        chain = prompt | llm.bind_tools(tools)
        result = safe_llm_invoke(chain, state["messages"], "Business Analyst")

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "business_report": report,
        }

    return business_node
