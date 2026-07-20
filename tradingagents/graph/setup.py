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
from .debate import build_debate


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
        # Build the analyst chain from the selection. Each spec carries its
        # own agent factory, tool node (or None for the in-node Sentiment
        # analyst), and report key.
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

        # --- Research debate: simultaneous rounds, Bull vs Bear ---
        # Order-neutral: both advocates fan out per round (operator.add reducer
        # on research_debate_turns), collector fans in, round-count conditional
        # routes to the synthesizer (Research Manager) after max_rounds.
        build_debate(
            workflow,
            entry_from="Report Digest",
            advocates={"Bull Researcher": "bull", "Bear Researcher": "bear"},
            synthesizer="Research Manager",
            turns_key="research_debate_turns",
            max_rounds=self.conditional_logic.max_debate_rounds,
        )

        # --- Risk debate: simultaneous rounds, Aggressive vs Conservative ---
        # Synthesizer = Neutral (pure synthesizer; offloads the PM, which
        # consumes Neutral's risk_synthesis instead of raw advocate turns).
        build_debate(
            workflow,
            entry_from="Research Manager",
            advocates={"Aggressive Analyst": "aggressive", "Conservative Analyst": "conservative"},
            synthesizer="Neutral Analyst",
            turns_key="risk_debate_turns",
            max_rounds=self.conditional_logic.max_risk_discuss_rounds,
        )

        workflow.add_edge("Neutral Analyst", "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", END)

        return workflow
