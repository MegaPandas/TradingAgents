"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    chat_prompt_messages,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

_RM_SYSTEM = """ROLE
You are the Research Manager. Adjudicate the bull/bear debate and issue the rating and the strategic actions the trader will act on.

SCOPE
1. Weigh which side the evidence supports; reserve Hold for genuinely balanced evidence.
2. Convert the verdict into the 5-tier rating: Buy / Overweight / Hold / Underweight / Sell.
3. Give the trader concrete strategic actions consistent with the rating (direction + conviction + the report evidence driving it).

HARD CONSTRAINTS
- Ground the rating in cited points from the debate; introduce no new analysis.
- Strategic actions must be implementable and reference the evidence.

OUTPUT
- recommendation (one of Buy / Overweight / Hold / Underweight / Sell)
- rationale (which arguments carried, cited)
- strategic_actions (concrete, evidence-anchored)"""


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")
        investment_debate_state = state["investment_debate_state"]
        report_digest = state.get("report_digest") or ""

        prompt_template = chat_prompt_messages(
            _RM_SYSTEM, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=(
                f"ANALYST REPORT DIGEST (pre-compressed):\n{report_digest}\n\n"
                f"DEBATE HISTORY:\n{history}\n\n"
                + get_language_instruction()
            ),
        )
        # Format to message list — invoke_structured_or_freetext expects
        # PromptValue | str | list[BaseMessage], not ChatPromptTemplate.
        formatted = prompt_template.format_messages(messages=state["messages"])

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
