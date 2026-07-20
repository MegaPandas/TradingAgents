"""Bear Researcher — simultaneous-debate advocate for the short/avoid case.

Mirror of bull_researcher with sides swapped. See graph/debate.py for topology.
"""

from tradingagents.agents.utils.agent_utils import (
    chat_prompt_messages,
    collect_analyst_reports,
    format_fair_value_block,
    get_instrument_context_from_state,
    get_language_instruction,
    safe_llm_invoke,
)
from tradingagents.agents.utils.debate_helpers import next_round_for_side, render_transcript

_SIDE = "bear"
_OPPONENT = "bull"
_SIDES = ("bull", "bear")


_BEAR_SYSTEM = """ROLE
You are the Bear Researcher — a partisan advocate for the SHORT/AVOID case. Your opponent is the Bull Researcher.

ABSOLUTE OUTPUT CONTRACT (every response must obey these):
1. Your output ALWAYS begins with the literal line `**Round {my_round}**` on its own line. Non-negotiable; it identifies which round the supervisor records.
2. Output ONLY the body for THIS round. No summary, no verdict, no probability blend, no stop-loss — those belong to the Research Manager, NOT you.
3. Round 1 — OPENING. Write:
   - One sentence: core bear thesis.
   - 3-5 cited evidence points (name the source analyst + the number).
   - A short "Acknowledged upside" line (1-2 items).
4. Round 2+ — DELTA-ONLY. Opponent's turn is in the transcript. You MUST:
   - Pick a specific claim from `Re: R{my_round-1}-{opp}` and rebut it, OR
   - Introduce new evidence the opponent has not yet seen.
   - Do NOT restate your R1. No verdict, no probability blend.

HARD CONSTRAINTS
- Cite the source analyst + number for every claim.
- Disagree on interpretation, never on facts.
- If opponent has no turn yet (transcript shows `<no turn>`), treat as opening and do NOT fabricate claims."""


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        turns = state.get("research_debate_turns", [])
        my_round = next_round_for_side(turns, _SIDE)
        transcript = render_transcript(turns, _SIDES)
        report_digest = state.get("report_digest") or collect_analyst_reports(state)
        instrument_context = get_instrument_context_from_state(state)
        fv_context = format_fair_value_block(state)

        round_directive = (
            "Round 1 (OPENING). No opponent turn exists yet. Write the opening position following the OUTPUT CONTRACT; "
            "no verdict, no probability blend, no stop-loss."
            if my_round == 1
            else
            f"Round {my_round} (DELTA-ONLY). Opponent's turn is in the transcript. "
            f"You MUST (a) rebut a specific claim from `Re: R{my_round-1}-{_OPPONENT}`, "
            f"or (b) introduce new evidence. "
            f"No restatement of your R1. No verdict. No probability blend. No stop-loss."
        )

        data_block = (
            f"{fv_context}\n\n"
            f"ANALYST REPORT DIGEST:\n{report_digest}\n\n"
            f"DEBATE TRANSCRIPT (canonical; opponent = {_OPPONENT}):\n{transcript}\n\n"
            f"YOUR ROUND: {my_round}\n"
            f"INSTRUCTION FOR THIS ROUND: {round_directive}\n\n"
            + get_language_instruction()
        )

        system_message = _BEAR_SYSTEM.replace("{my_round}", str(my_round))

        prompt_template = chat_prompt_messages(
            system_message, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = safe_llm_invoke(llm, formatted, "Bear Researcher")

        return {"research_debate_turns": [{"round": my_round, "side": _SIDE, "delta": response.content}]}

    return bear_node
