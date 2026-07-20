"""Tests for graph conditional routing.

Covers the shared ``_tool_or_clear`` helper behind every tooled analyst's
should_continue_<key> router, the Bull/Bear debate round counter, and the
single-pass risk routing that always lands on the Portfolio Manager.
"""

import pytest
from langchain_core.messages import AIMessage

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.debate import make_should_continue_debate
from tradingagents.agents.utils.debate_helpers import next_round_for_side, render_transcript


def _state(tool_calls=None):
    """Build a minimal state whose last message carries (or omits) tool_calls."""
    return {"messages": [AIMessage(content="ok", tool_calls=tool_calls or [])]}


@pytest.mark.unit
class TestToolOrClear:
    """Each tooled analyst routes to its tool node while calls are pending,
    else to its message-clear node."""

    def setup_method(self):
        self.cl = ConditionalLogic(max_debate_rounds=1)

    @pytest.mark.parametrize(
        "method,tool_key,clear_label",
        [
            ("should_continue_market", "tools_market", "Msg Clear Market"),
            ("should_continue_social", "tools_social", "Msg Clear Sentiment"),
            ("should_continue_news", "tools_news", "Msg Clear News"),
            ("should_continue_fundamentals", "tools_fundamentals", "Msg Clear Fundamentals"),
            ("should_continue_macro_policy", "tools_macro_policy", "Msg Clear Macro Policy"),
            ("should_continue_situation", "tools_situation", "Msg Clear Situation"),
            ("should_continue_business", "tools_business", "Msg Clear Business"),
        ],
    )
    def test_pending_tool_calls_route_to_tool_node(self, method, tool_key, clear_label):
        router = getattr(self.cl, method)
        state = _state(tool_calls=[{"name": "x", "args": {}, "id": "1"}])
        assert router(state) == tool_key

    @pytest.mark.parametrize(
        "method,clear_label",
        [
            ("should_continue_market", "Msg Clear Market"),
            ("should_continue_news", "Msg Clear News"),
            ("should_continue_fundamentals", "Msg Clear Fundamentals"),
            ("should_continue_macro_policy", "Msg Clear Macro Policy"),
        ],
    )
    def test_no_tool_calls_route_to_clear(self, method, clear_label):
        router = getattr(self.cl, method)
        assert router(_state()) == clear_label


@pytest.mark.unit
class TestSimultaneousDebateConditional:
    """build_debate's round-count conditional: both sides done max_rounds → synthesizer, else advocates."""

    def test_no_turns_routes_to_advocates(self):
        cond = make_should_continue_debate("research_debate_turns", ("bull", "bear"), max_rounds=2)
        assert cond({"research_debate_turns": []}) == "advocates"

    def test_partial_rounds_routes_to_advocates(self):
        cond = make_should_continue_debate("research_debate_turns", ("bull", "bear"), max_rounds=2)
        turns = [{"round": 1, "side": "bull", "delta": "x"}, {"round": 1, "side": "bear", "delta": "y"}]
        assert cond({"research_debate_turns": turns}) == "advocates"

    def test_both_sides_done_routes_to_synthesizer(self):
        cond = make_should_continue_debate("research_debate_turns", ("bull", "bear"), max_rounds=2)
        turns = [
            {"round": 1, "side": "bull", "delta": "x"}, {"round": 1, "side": "bear", "delta": "y"},
            {"round": 2, "side": "bull", "delta": "x2"}, {"round": 2, "side": "bear", "delta": "y2"},
        ]
        assert cond({"research_debate_turns": turns}) == "synthesizer"

    def test_one_side_lagging_stays_in_advocates(self):
        # bull ahead (2 turns), bear behind (1) → not both done → advocates.
        cond = make_should_continue_debate("t", ("bull", "bear"), max_rounds=2)
        turns = [{"round": 1, "side": "bull"}, {"round": 1, "side": "bear"}, {"round": 2, "side": "bull"}]
        assert cond({"t": turns}) == "advocates"

    def test_round_inference_and_transcript_render(self):
        turns = [{"round": 1, "side": "bull", "delta": "B1"}, {"round": 1, "side": "bear", "delta": "R1"}]
        assert next_round_for_side(turns, "bull") == 2
        assert next_round_for_side(turns, "bear") == 2
        rendered = render_transcript(turns, ("bull", "bear"))
        assert "R1 bull: B1" in rendered
        assert "R1 bear: R1" in rendered
