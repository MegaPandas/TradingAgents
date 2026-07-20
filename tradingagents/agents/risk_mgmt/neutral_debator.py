"""Neutral Risk Analyst — pure synthesizer of the risk debate.

Reads the canonical risk-debate transcript (Aggressive vs Conservative) and
distills it into a structured `risk_synthesis` (probability-weighted EV +
scenario probabilities + balanced sizing) for the Portfolio Manager. NOT an
advocate — introduces no partisan stance. This is the risk-chain synthesizer,
parallel to the Research Manager in the research chain; it exists so the
overloaded PM consumes a compact summary instead of the raw advocate turns.
"""

from tradingagents.agents.utils.agent_utils import (
    chat_prompt_messages,
    format_fair_value_block,
    format_price_context,
    get_instrument_context_from_state,
    get_language_instruction,
    safe_llm_invoke,
)
from tradingagents.agents.utils.debate_helpers import render_transcript

_SIDES = ("aggressive", "conservative")

_NEUTRAL_SYSTEM = """ROLE
You are the Neutral Risk Analyst. Your job is to SYNTHESIZE the Aggressive and Conservative cases into a probability-weighted expected-value view and a balanced sizing — NOT to argue a side.

SCOPE
1. Compute a probability-weighted expected return across the upside/downside scenarios.
2. State where each side is right and where each over-reaches (cite the SPECIFIC claim).
3. Recommend a balanced position sizing.

HARD CONSTRAINTS
- Show the EV math. Cite both sides.
- PROBABILITIES MUST SUM TO 1.00 (state the bear/base/bull probabilities; note any rounding residual).
- EXPECTED VALUE must equal sum(p × r) exactly — quote that number, not a rounded approximation.
- Cross-citation must NAME the specific claim (e.g. "Aggressive cited PEG=1.25 — Conservative countered FY25 -19%...").
- PROBABILITY METHOD: weights reflect forward-looking uncertainty, not the magnitude of already-priced facts (already-priced bad news ⇒ LOWER bear probability).
- ANCHOR BRACKET: bull target ≥ FV normalized median +15%, bear ≤ FV Justified PB / Residual Income low, base ≈ FV median ±5%.

OUTPUT
- Bear / base / bull probabilities (summing to 1.00) + each scenario's target return
- Probability-weighted expected value (show p × r sum)
- Balanced sizing
- The strongest point from each side (cited) + where each over-reaches
- Scenario brackets vs FV (cite which constraint each satisfies)"""


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        turns = state.get("risk_debate_turns", [])
        transcript = render_transcript(turns, _SIDES)
        instrument_context = get_instrument_context_from_state(state)
        price_context = format_price_context(state)
        fv_context = format_fair_value_block(state)

        data_block = (
            f"{price_context}\n{fv_context}\n\n"
            f"RISK DEBATE TRANSCRIPT (Aggressive vs Conservative, canonical):\n{transcript}\n\n"
            + get_language_instruction()
        )

        prompt_template = chat_prompt_messages(
            _NEUTRAL_SYSTEM, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = safe_llm_invoke(llm, formatted, "Neutral Analyst")

        return {"risk_synthesis": response.content}

    return neutral_node
