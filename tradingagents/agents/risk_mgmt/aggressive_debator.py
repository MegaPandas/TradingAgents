"""Aggressive Risk Analyst — simultaneous-debate advocate for the upside case.

Round-aware: reads the risk-debate transcript, infers its round, writes a
delta. Quantifies upside target + probability of UNPRICED upside + sizing tilt.
"""

from tradingagents.agents.utils.agent_utils import (
    _NO_FABRICATED_BASE_RATES,
    chat_prompt_messages,
    collect_analyst_reports,
    format_fair_value_block,
    format_price_context,
    get_instrument_context_from_state,
    get_language_instruction,
    safe_llm_invoke,
)
from tradingagents.agents.utils.debate_helpers import next_round_for_side, render_transcript

_SIDE = "aggressive"
_OPPONENT = "conservative"
_SIDES = ("aggressive", "conservative")


_AGGRESSIVE_SYSTEM = f"""ROLE
You are the Aggressive Risk Analyst — partisan advocate for the UPSIDE (bull) case. Your opponent is the Conservative Risk Analyst.

ABSOLUTE OUTPUT CONTRACT (every response must obey):
1. Your output ALWAYS begins with the literal line `**Round {{my_round}}**` on its own line.
2. Output ONLY the body for THIS round. No summary, no combined verdict, no "we recommend" — those belong to Neutral, NOT you.
3. Round 1 — OPENING (no opponent turn yet). Write:
   - Bull-case price target, catalysts, estimated probability of UNPRICED upside.
   - Sizing tilt toward upside capture.
   - A short "Concession" line (1-2 items you agree Conservative on).
4. Round 2+ — DELTA-ONLY. Opponent's turn is in the transcript. You MUST:
   - Pick a specific claim from `Re: R{{my_round-1}}-{_OPPONENT}` and rebut it, OR
   - Introduce new upside evidence the opponent has not yet seen.
   - Do NOT restate your R1. No combined probability, no verdict.

HARD CONSTRAINTS
- Probability must reflect the chance of UNPRICED forward-looking upside (catalysts the market has NOT yet priced), not the magnitude of already-known positives.
- For each bull argument answer "why hasn't the market priced this yet?" — if you cannot, lower the probability.
- Bull target ≥ FV normalized median + 15% (positive skew above anchor).
- {_NO_FABRICATED_BASE_RATES}"""


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        turns = state.get("risk_debate_turns", [])
        my_round = next_round_for_side(turns, _SIDE)
        transcript = render_transcript(turns, _SIDES)
        instrument_context = get_instrument_context_from_state(state)
        price_context = format_price_context(state)
        fv_context = format_fair_value_block(state)
        report_digest = state.get("report_digest") or collect_analyst_reports(state)

        round_directive = (
            "Round 1 (OPENING). No opponent turn exists yet. Write upside target + probability + sizing; "
            "no combined probability, no verdict."
            if my_round == 1
            else
            f"Round {my_round} (DELTA-ONLY). Opponent's turn is in the transcript. "
            f"You MUST (a) rebut a specific claim from `Re: R{{my_round-1}}-{_OPPONENT}`, "
            f"or (b) introduce new upside evidence. "
            f"No restatement of your R1. No combined probability, no verdict."
        )

        data_block = (
            f"{price_context}\n{fv_context}\n\n"
            f"ANALYST REPORT DIGEST:\n{report_digest}\n\n"
            f"DEBATE TRANSCRIPT (canonical; opponent = {_OPPONENT}):\n{transcript}\n\n"
            f"YOUR ROUND: {my_round}\n"
            f"INSTRUCTION FOR THIS ROUND: {round_directive}\n\n"
            + get_language_instruction()
        )

        system_message = _AGGRESSIVE_SYSTEM.replace("{my_round}", str(my_round))

        prompt_template = chat_prompt_messages(
            system_message, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = safe_llm_invoke(llm, formatted, "Aggressive Analyst")

        return {"risk_debate_turns": [{"round": my_round, "side": _SIDE, "delta": response.content}]}

    return aggressive_node
