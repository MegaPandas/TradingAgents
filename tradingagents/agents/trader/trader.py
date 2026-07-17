"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        # Trader sets entry/stop around the actual current price. Inject the
        # distinct realtime-intraday vs settled-daily close so it doesn't anchor
        # levels on a stale or invented number.
        from tradingagents.agents.utils.agent_utils import format_price_context
        price_ctx = format_price_context(state)
        close_line = price_ctx.rstrip() if price_ctx else (
            "- PRICE: UNAVAILABLE — fall back to the most recent close cited "
            "in any analyst report and state the source"
        )

        # Report digest so the trader can cite Technical (ATR / levels) and
        # Valuation (fair value) evidence when setting entry/exit/sizing.
        report_digest = state.get("report_digest") or "(report_digest not populated)"

        messages = [
            {
                "role": "system",
                "content": (
                    "ROLE\n"
                    "You are the Trader. Convert the Research Manager's plan into an executable transaction: direction, price levels, sizing, and execution method.\n\n"
                    "SCOPE\n"
                    "1. Action — Buy / Sell / Hold, consistent with the research rating.\n"
                    "2. Levels — entry (or exit) zone and stop-loss / invalidation, drawn from the Technical and Valuation reports.\n"
                    "3. Sizing — position size as a percent of portfolio, scaled to conviction and to ATR-derived risk.\n"
                    "4. Execution — scaling (single vs staged), timing trigger, and the condition that voids the trade.\n\n"
                    "HARD CONSTRAINTS\n"
                    "- Every level must trace to a cited number in the Technical or Valuation report. No invented round numbers.\n"
                    "- If the rating is Hold, still state the condition that would trigger an entry.\n\n"
                    "OUTPUT\n"
                    "- action, reasoning (cited)\n"
                    "- entry_price, stop_loss, position_sizing (required for Buy/Sell)\n"
                    "- execution notes (scaling / trigger / void condition)"
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research Manager's investment plan for {company_name}: {instrument_context}\n\n"
                    f"AVAILABLE CONTEXT:\n{close_line}\n\n"
                    f"ANALYST REPORT DIGEST (cite specific points by section when setting levels):\n{report_digest}\n\n"
                    f"Research Manager's investment plan:\n{investment_plan}\n\n"
                    "Produce the transaction proposal."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
