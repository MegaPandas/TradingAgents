"""Agent state for the TradingAgents graph.

The two debates (research + risk) use a **simultaneous-rounds** transcript:
each advocate appends a turn dict ``{"round": int, "side": str, "delta": str}``
to a shared list with an ``operator.add`` reducer, so both advocates can write
in parallel without the nested-TypedDict reducer collision that forced the old
separate-key workaround. Round count is derived from the transcript (each
advocate infers its round from its own turn count) — no external counter.
"""

import operator
from typing import Annotated

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State flowing through every node of the trading graph."""

    # --- Identity ---
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    asset_type: Annotated[str, "Asset type under analysis such as stock or crypto"]
    instrument_context: Annotated[str, "Deterministic ticker identity resolved at run start"]
    trade_date: Annotated[str, "What date we are trading at"]

    # --- Pre-fetched data from Propagation node ---
    latest_close: Annotated[float | None, "Settled daily close (qfq) from sina minute bars"]
    latest_close_settled_at: Annotated[str | None, "Timestamp of settled close"]
    latest_close_realtime: Annotated[float | None, "Realtime intraday close (raw) from sina minute bars"]
    latest_close_realtime_at: Annotated[str | None, "Timestamp of realtime close"]
    news_block: Annotated[str | None, "Pre-fetched 7-day news block shared by News + Sentiment"]
    report_digest: Annotated[str, "Deterministic digest of all analyst reports"]
    fair_value_block: Annotated[str, "Pre-computed deterministic 5-model fair value (no LLM) — PM anchor"]

    # --- Analyst reports ---
    market_report: Annotated[str, "Report from the Market Analyst"]
    sentiment_report: Annotated[str, "Report from the Sentiment Analyst"]
    news_report: Annotated[str, "Report from the News Analyst"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst"]
    macro_policy_report: Annotated[str, "Report from the Macro & Policy Analyst"]
    business_report: Annotated[str, "Report from the Business Analyst (segment breakdown)"]
    situation_report: Annotated[str, "Report from the Situation Analyst (price journey narrative)"]

    # --- Research debate (simultaneous rounds, order-neutral) ---
    # Each Bull/Bear advocate appends {"round", "side", "delta"} per round.
    # operator.add lets both append concurrently; the transcript is the single
    # source the Research Manager reads (no narrative history re-paste).
    research_debate_turns: Annotated[list[dict], operator.add]
    investment_plan: Annotated[str, "Research Manager's structured synthesis of the debate"]

    # --- Risk debate (simultaneous rounds, order-neutral) ---
    # Each Aggressive/Conservative advocate appends a turn per round, same shape.
    risk_debate_turns: Annotated[list[dict], operator.add]
    risk_synthesis: Annotated[str, "Neutral's structured synthesis (EV + scenario probabilities + sizing)"]

    final_trade_decision: Annotated[str, "Final decision made by the Portfolio Manager"]
    past_context: Annotated[str, "Memory log context injected at run start (same-ticker decisions + cross-ticker lessons)"]
