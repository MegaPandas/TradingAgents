# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def _tool_or_clear(self, state: AgentState, tool_key: str, clear_label: str) -> str:
        """Return the tool node if the last message has pending tool calls,
        otherwise the message-clear node. Shared by every tooled analyst."""
        last_message = state["messages"][-1]
        return tool_key if last_message.tool_calls else clear_label

    def should_continue_market(self, state: AgentState):
        return self._tool_or_clear(state, "tools_market", "Msg Clear Market")

    def should_continue_social(self, state: AgentState):
        """Sentiment analyst wire key kept as ``social`` for saved-config
        back-compat; the clear label uses the v0.2.5 rename."""
        return self._tool_or_clear(state, "tools_social", "Msg Clear Sentiment")

    def should_continue_news(self, state: AgentState):
        return self._tool_or_clear(state, "tools_news", "Msg Clear News")

    def should_continue_fundamentals(self, state: AgentState):
        return self._tool_or_clear(state, "tools_fundamentals", "Msg Clear Fundamentals")

    def should_continue_macro_policy(self, state: AgentState):
        return self._tool_or_clear(state, "tools_macro_policy", "Msg Clear Macro Policy")

    def should_continue_situation(self, state: AgentState):
        return self._tool_or_clear(state, "tools_situation", "Msg Clear Situation")

    def should_continue_business(self, state: AgentState):
        return self._tool_or_clear(state, "tools_business", "Msg Clear Business")

    # NOTE: research + risk debates are now wired by graph/debate.build_debate
    # (simultaneous rounds, round-count conditional). The max_debate_rounds /
    # max_risk_discuss_rounds stored here feed _run_signature (checkpoint
    # invalidation) and are passed to build_debate from setup_graph.
