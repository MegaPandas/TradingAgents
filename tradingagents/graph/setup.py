# TradingAgents/graph/setup.py

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
)
from tradingagents.agents.analysts.macro_policy_analyst import (
    create_macro_policy_analyst,
)
from tradingagents.agents.analysts.situation_analyst import (
    create_situation_analyst,
)
from tradingagents.agents.analysts.business_analyst import (
    create_business_analyst,
)
from tradingagents.agents.utils.agent_states import AgentState

from .analyst_execution import ANALYST_NODE_SPECS, AnalystNodeSpec, build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst (wire key kept as "social")
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
                - "macro_policy": Macro & Policy analyst (P2 — opt-in)
        """
        # The Macro & Policy analyst has no tool-calling (data is pre-fetched
        # in the node), so its empty ToolNode and one-line conditional live in
        # the registries owned by `trading_graph.GraphSetup.__init__` callers.
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
            "macro_policy": lambda: create_macro_policy_analyst(self.quick_thinking_llm),
            "situation": lambda: create_situation_analyst(self.quick_thinking_llm),
            "business": lambda: create_business_analyst(self.quick_thinking_llm),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            # Only register a ToolNode when the analyst actually has one
            # (Sentiment pre-fetches in-node, so spec.tool_node is None).
            if spec.tool_node is not None:
                workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Insert the deterministic analyst-report digest between the last
        # analyst's message-clear node and Bull Researcher (D1 — pipeline
        # dedup). Every downstream debate/decision node reads `report_digest`
        # instead of the four full reports, so a single shared doc caps the
        # prompt size and prevents analysts from drifting apart.
        def report_digest_node(state):
            from tradingagents.agents.utils.agent_utils import (
                build_digest_for_state,
            )

            return {"report_digest": build_digest_for_state(state)}

        workflow.add_node("Report Digest", report_digest_node)

        # Define edges
        # Start with the first analyst
        workflow.add_edge(START, plan.specs[0].agent_node)

        # Connect analysts in sequence
        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            # Add conditional edges for current analyst.  When the analyst
            # has no ToolNode (e.g. Sentiment), the only branch is the
            # clear-node — pass it as a single-element list rather than
            # omitting path_map, which langgraph would treat as "unbounded".
            branch_paths = (
                [current_clear] if current_tools is None
                else [current_tools, current_clear]
            )
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                branch_paths,
            )
            if current_tools is not None:
                workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst, or to Report Digest if this is the last analyst
            if i < len(plan.specs) - 1:
                workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(current_clear, "Report Digest")

        workflow.add_edge("Report Digest", "Bull Researcher")

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Aggressive Analyst")
        workflow.add_edge("Research Manager", "Conservative Analyst")
        # Aggressive and Conservative run in parallel after Research Manager.
        # Each writes to a SEPARATE state key (aggressive_risk_argument /
        # conservative_risk_argument) — no reducer collision on the nested
        # TypedDict.  Each gets ONE turn, then routes directly to Neutral.
        workflow.add_edge("Aggressive Analyst", "Neutral Analyst")
        workflow.add_edge("Conservative Analyst", "Neutral Analyst")
        # Neutral synthesises both arguments into risk_debate_state, then
        # routes to Portfolio Manager.
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            RISK_ANALYSIS_PATH_MAP,
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
