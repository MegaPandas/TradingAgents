from tradingagents.agents.utils.agent_utils import (
    chat_prompt_messages,
    format_fair_value_block,
    format_price_context,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.math_validators import (
    validate_expected_value,
    validate_probability_sum,
)

_NEUTRAL_SYSTEM = """ROLE
You are the Neutral Risk Analyst. Weigh both scenarios into an expected-value view and a balanced sizing.

SCOPE
1. Compute a probability-weighted expected return across the upside and downside scenarios.
2. State where each side is right.
3. Add a diversification / correlation note for the position.

HARD CONSTRAINTS
- Show the expected-value math. Cite both sides.
- PROBABILITIES MUST SUM TO 1.00: when assigning bear / base / bull probabilities, normalize them so they sum to 100% (or state the rounding residual explicitly). A "40% + 30% + 30%" tally without a note that the residual is 0 is a math error — re-do. Before publishing, mentally run `validate_probability_sum([bear_p, base_p, bull_p])` — sum should be 1.0 +/- 0.01.
- EXPECTED VALUE must equal sum(p * r) exactly. Mentally compute `validate_expected_value([(p_bear, r_bear), (p_base, r_base), (p_bull, r_bull)])` and quote that number, not a rounded or directional approximation.
- When citing each side, name the SPECIFIC claim being acknowledged (e.g. "Aggressive cited PEG=1.25 — Conservative countered that FY25 profit -19% means revenue growth isn't converting; both have ground"). Without naming the specific point, the cross-citation is rhetorical.
- PROBABILITY METHOD: your scenario weights must reflect forward-looking uncertainty, not inherit the conservative debator's bias toward confirmed lagging data. If the bear case rests on "Q1 earnings -55%" (already priced into the current PE), that fact does NOT justify a 50% bear probability — it justifies a LOWER bear probability because the bad news is already in the price. Weigh each scenario by the probability of UNPRICED marginal change, not by the magnitude of already-known facts.
- **ANCHOR BRACKET (Fix 1)**: scenario targets MUST bracket the deterministic fair-value range (in the DATA block below). Bull case target >= FV normalized median + 15% (positive skew); Bear case target <= FV Justified PB / Residual Income low (negative skew). Base case target approx FV normalized median +/- 5%. If your scenario targets stray outside this bracket, you are constructing an anchor instead of synthesising the debate — re-anchor to FV.

OUTPUT
- Probability-weighted expected value (show p*r sum)
- Balanced sizing
- The strongest point from each side (cited)
- Scenario brackets vs FV normalized median (cite which constraint each scenario satisfies)"""


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        aggressive_arg = state.get("aggressive_risk_argument", "")
        conservative_arg = state.get("conservative_risk_argument", "")

        report_digest = state.get("report_digest") or (
            state.get("market_report", "") + state.get("sentiment_report", "") +
            state.get("news_report", "") + state.get("fundamentals_report", "") +
            state.get("macro_policy_report", "") + state.get("business_report", "") +
            state.get("situation_report", "")
        )
        instrument_context = get_instrument_context_from_state(state)
        price_context = format_price_context(state)
        fv_context = format_fair_value_block(state)

        data_block = (
            f"{price_context}\n{fv_context}\n\n"
            f"ANALYST REPORT DIGEST:\n{report_digest}\n\n"
            f"AGGRESSIVE ARGUMENT (bull case):\n{aggressive_arg}\n\n"
            f"CONSERVATIVE ARGUMENT (bear case):\n{conservative_arg}\n\n"
            + get_language_instruction()
        )

        prompt_template = chat_prompt_messages(
            _NEUTRAL_SYSTEM, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = llm.invoke(formatted)

        argument = f"Neutral Analyst: {response.content}"
        history = aggressive_arg + "\n" + conservative_arg + "\n" + argument

        new_risk_debate_state = {
            "history": history,
            "aggressive_history": aggressive_arg,
            "conservative_history": conservative_arg,
            "neutral_history": argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": aggressive_arg,
            "current_conservative_response": conservative_arg,
            "current_neutral_response": argument,
            "judge_decision": "",
            "count": 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
