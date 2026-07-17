# TradingAgents/graph/propagation.py

from typing import Any

import pandas as pd

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.agents.utils.news_data_tools import get_news


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.
        """
        state = {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            # Separate keys for parallel Aggressive+Conservative (no reducer
            # possible on nested TypedDict — each debator writes its own key).
            "aggressive_risk_argument": "",
            "conservative_risk_argument": "",
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "news_block": None,
            "fair_value_block": "",
        }

        # P1 dedup (D3): pre-fetch the news block once so the News and Sentiment
        # analysts share a single fetch instead of each invoking get_news.
        # Pre-fetch failure must not abort propagation; analysts fall back.
        try:
            end = state["trade_date"]
            start = (pd.Timestamp(end) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            state["news_block"] = get_news.func(state["company_of_interest"], start, end)
        except Exception:
            state["news_block"] = None

        # Pre-compute deterministic fair value (5 academic models, no LLM).
        # The Portfolio Manager MUST anchor to this, not debate scenarios.
        # calculate_fair_value auto-detects cyclical troughs (TTM EPS <
        # 85% of FY EPS) and emits a normalized-EPS range automatically —
        # same code path the Fundamentals Analyst tool uses, so both PM
        # (via this pre-fetch) and Fundamentals (via get_fair_value tool)
        # see identical output.  No divergence.
        # Pre-fetch failure must not abort propagation; PM falls back.
        try:
            from tradingagents.dataflows.fair_value import calculate_fair_value
            state["fair_value_block"] = calculate_fair_value(state["company_of_interest"])
        except Exception:
            state["fair_value_block"] = ""

        # Pre-fetch TWO price snapshots so downstream agents never confuse a
        # settled daily close with a live intraday print:
        #   * realtime minute close (adjust="")  — today's intraday, pre-settlement
        #   * settled daily close  (adjust="qfq") — last trading day's official close
        # sina's qfq endpoint returns NaN for today's incomplete bars (it can't
        # dividend-adjust an unsettled bar), so during market hours ONLY the raw
        # minute endpoint exposes the live price. After close both converge.
        state["latest_close_realtime"] = None
        state["latest_close_realtime_at"] = None
        state["latest_close"] = None
        state["latest_close_settled_at"] = None
        try:
            from tradingagents.dataflows.akshare_data import _no_proxy
            from tradingagents.dataflows.symbol_utils import cn_sina_symbol, cn_code6
            import akshare as ak
            ticker = state["company_of_interest"]
            sina = cn_sina_symbol(ticker)
            if not sina:
                code6 = cn_code6(ticker) or ticker.upper()
                sina = f"sz{code6}"
            symbol = sina

            # 1) Realtime intraday (raw, no adjustment) — today's live price.
            with _no_proxy():
                rt = ak.stock_zh_a_minute(symbol=symbol, period="60", adjust="")
            if rt is not None and not rt.empty and "close" in rt.columns:
                rt = rt.dropna(subset=["close"])
                if not rt.empty:
                    state["latest_close_realtime"] = float(rt["close"].iloc[-1])
                    state["latest_close_realtime_at"] = str(rt["day"].iloc[-1])

            # 2) Settled daily close (qfq) — last official close; for valuation
            #    anchoring and 52-week / historical comparisons.
            with _no_proxy():
                qf = ak.stock_zh_a_minute(symbol=symbol, period="60", adjust="qfq")
            if qf is not None and not qf.empty and "close" in qf.columns:
                qf = qf.dropna(subset=["close"])
                if not qf.empty:
                    state["latest_close"] = float(qf["close"].iloc[-1])
                    state["latest_close_settled_at"] = str(qf["day"].iloc[-1])

            # Prefer realtime for "current price"; fall back to settled.
            if state["latest_close_realtime"] is None:
                state["latest_close_realtime"] = state["latest_close"]
                state["latest_close_realtime_at"] = state["latest_close_settled_at"]
        except Exception:
            pass

        return state

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
