"""Conservative Risk Analyst — simultaneous-debate advocate for the downside case.

Mirror of aggressive_debator with sides swapped. Quantifies downside target +
probability of UNPRICED deterioration + capital-preservation sizing.
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

_SIDE = "conservative"
_OPPONENT = "aggressive"
_SIDES = ("aggressive", "conservative")


_CONSERVATIVE_SYSTEM = f"""ROLE
You are the Conservative Risk Analyst — partisan advocate for the DOWNSIDE (bear) case. Your opponent is the Aggressive Risk Analyst.

ABSOLUTE OUTPUT CONTRACT (every response must obey):
1. Your output ALWAYS begins with the literal line `**Round {{my_round}}**` on its own line.
2. Output ONLY the body for THIS round. No summary, no combined verdict, no "we recommend" — those belong to Neutral, NOT you.
3. Round 1 — OPENING (no opponent turn yet). Write:
   - Bear-case price target, triggers, probability of UNPRICED deterioration, max-drawdown path.
   - Capital-preservation sizing chain (base × trend × earnings-uncertainty, shown step-by-step).
   - A short "Concession" line (1-2 items you agree Aggressive on).
4. Round 2+ — DELTA-ONLY. Opponent's turn is in the transcript. You MUST:
   - Pick a specific claim from `Re: R{{my_round-1}}-{_OPPONENT}` and rebut it, OR
   - Introduce new downside evidence the opponent has not yet seen.
   - Do NOT restate your R1. No combined probability, no verdict.

HARD CONSTRAINTS
- Probability must reflect UNPRICED forward-looking deterioration, NOT the magnitude of already-priced bad news.
- For each bear argument answer "why hasn't the market discounted this yet?" — if you cannot, lower the probability.
- Any multiplicative sizing chain must be shown step-by-step with the product quoted.
- Bear target ≤ FV Justified PB / Residual Income low (negative skew).
- {_NO_FABRICATED_BASE_RATES}"""


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        turns = state.get("risk_debate_turns", [])
        my_round = next_round_for_side(turns, _SIDE)
        transcript = render_transcript(turns, _SIDES)
        instrument_context = get_instrument_context_from_state(state)
        price_context = format_price_context(state)
        fv_context = format_fair_value_block(state)
        report_digest = state.get("report_digest") or collect_analyst_reports(state)

        round_directive = (
            "Round 1 (OPENING). No opponent turn exists yet. Write downside target + probability + sizing chain; "
            "no combined probability, no verdict."
            if my_round == 1
            else
            f"Round {my_round} (DELTA-ONLY). Opponent's turn is in the transcript. "
            f"You MUST (a) rebut a specific claim from `Re: R{{my_round-1}}-{_OPPONENT}`, "
            f"or (b) introduce new downside evidence. "
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

        system_message = _CONSERVATIVE_SYSTEM.replace("{my_round}", str(my_round))

        prompt_template = chat_prompt_messages(
            system_message, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = safe_llm_invoke(llm, formatted, "Conservative Analyst")

        return {"risk_debate_turns": [{"round": my_round, "side": _SIDE, "delta": response.content}]}

    return conservative_node
