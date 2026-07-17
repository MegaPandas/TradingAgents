from tradingagents.agents.utils.agent_utils import (
    chat_prompt_messages,
    format_fair_value_block,
    get_instrument_context_from_state,
    get_language_instruction,
)

_BEAR_SYSTEM = """ROLE
You are the Bear Researcher. Build the strongest evidence-based short/avoid case from the analyst reports, and answer the bull case on its own terms.

SCOPE
1. State the core bear thesis in one sentence.
2. Give 3–5 supporting evidence points, each citing which analyst report and which number.
3. Address the bull's strongest points directly, with data — not assertion.
4. Acknowledge the genuine upside (credibility requires it).

HARD CONSTRAINTS
- Cite the source analyst report for every claim. Introduce no data not present in the analyst reports or tool outputs.
- Disagree on interpretation, never on facts.
- Begin your reply with the literal line `**Round N**` (replace N with this round's number, derivable from the history length above).
- Do NOT restate points already in the debate history; write ONLY (a) new evidence, (b) explicit rebuttals to specific opposing arguments (cite them), or (c) updated numbers. Restating your earlier argument is a wasted turn.
- If the opposing side has not yet spoken this round, focus on (a) and (c).

OUTPUT
- Thesis (one sentence)
- Evidence (3–5 cited points)
- Response to bull (point-by-point, cited)
- Acknowledged upside"""


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        report_digest = state.get("report_digest") or (
            state.get("market_report", "") + state.get("sentiment_report", "") +
            state.get("news_report", "") + state.get("fundamentals_report", "") +
            state.get("macro_policy_report", "") + state.get("business_report", "") +
            state.get("situation_report", "")
        )
        instrument_context = get_instrument_context_from_state(state)
        fv_context = format_fair_value_block(state)

        data_block = (
            fv_context +
            f"\n\nANALYST REPORT DIGEST:\n{report_digest}\n\n"
            f"DEBATE HISTORY:\n{history}\n\n"
            f"LAST BULL ARGUMENT:\n{current_response}\n\n"
            + get_language_instruction()
        )

        prompt_template = chat_prompt_messages(
            _BEAR_SYSTEM, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = llm.invoke(formatted)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
