from tradingagents.agents.utils.agent_utils import (
    _NO_FABRICATED_BASE_RATES,
    chat_prompt_messages,
    format_fair_value_block,
    format_price_context,
    get_instrument_context_from_state,
    get_language_instruction,
)

_AGGRESSIVE_SYSTEM = f"""ROLE
You are the Aggressive Risk Analyst. Quantify the upside scenario for the research manager's plan and challenge under-stated bull cases.

SCOPE
1. State the bull-case price target, the catalysts, and an estimated probability.
2. Recommend a sizing tilt toward upside capture.
3. Identify specifically where the conservative case under-states probability or magnitude (cited).

HARD CONSTRAINTS
- Assign probabilities to scenarios; cite data for each. No rhetorical cheerleading.
- Disagree on interpretation, never on facts.
- Begin your reply with the literal line `**Round N**` (replace N with this round's number, derivable from the history length above).
- Do NOT restate points already in the debate history; write ONLY (a) new evidence, (b) explicit rebuttals to specific opposing arguments (cite them), or (c) updated numbers. Restating your earlier argument is a wasted turn.
- **PARALLEL-DEBATE FORMAT (Fix 5)**: Aggressive and Conservative run in PARALLEL after the Research Manager. In Round 1 the opposing side has not yet spoken — `current_conservative_response` and `current_neutral_response` are empty placeholders. Do NOT fabricate specific Conservative arguments; instead, anticipate the strongest reasonable bear position and pre-empt it (e.g. "I expect Conservative will argue [X]; my response is [Y]"). In subsequent rounds when the opponent has actually spoken, switch to direct rebuttal.
- GROWTH-NAME OVERRIDE: when the fundamentals report shows revenue 3yr CAGR ≥ 25% or growth_quality_score ≥ 25/40, your bull case must engage with the specific bear-evidence cited (PEG value, TAM saturation, margin trajectory). "Growth justifies premium" without addressing the bear's specific points is not a sufficient rebuttal.
- PROBABILITY METHOD: your bull-case probability must reflect the chance of UNPRICED forward-looking upside — catalysts the market has NOT yet priced. A confirmed positive (export +95%, solid-state trial) that is already widely known does not raise the probability of further upside. Anchor your probability on: (a) what positive surprises lie ahead that consensus underestimates, (b) the gap between market expectations and your scenario, (c) historical base-rate of similar re-ratings.
- PRICING SELF-CHECK: for each bull argument, answer "why hasn't the market priced this yet?" — is the catalyst unpublished, under-reported by consensus, or already reflected in the current price? If you cannot explain why the market has NOT yet priced your upside, lower your probability — the market is not stupid.
- {_NO_FABRICATED_BASE_RATES}

OUTPUT
- Upside scenario (target, catalysts, probability)
- Sizing tilt
- Points where the conservative case is too cautious (cited)
- Bull target must be ≥ FV normalized median + 15% (positive skew above anchor)"""


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")
        instrument_context = get_instrument_context_from_state(state)
        price_context = format_price_context(state)
        fv_context = format_fair_value_block(state)

        report_digest = state.get("report_digest") or (
            state.get("market_report", "") + state.get("sentiment_report", "") +
            state.get("news_report", "") + state.get("fundamentals_report", "") +
            state.get("macro_policy_report", "") + state.get("business_report", "") +
            state.get("situation_report", "")
        )

        data_block = (
            f"{price_context}\n{fv_context}\n\n"
            f"ANALYST REPORT DIGEST:\n{report_digest}\n\n"
            f"DEBATE HISTORY:\n{history}\n\n"
            f"LAST CONSERVATIVE ARGUMENT:\n{current_conservative_response}\n\n"
            f"LAST NEUTRAL ARGUMENT:\n{current_neutral_response}\n\n"
            + get_language_instruction()
        )

        prompt_template = chat_prompt_messages(
            _AGGRESSIVE_SYSTEM, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = llm.invoke(formatted)

        argument = f"Aggressive Analyst: {response.content}"

        # Write to a SEPARATE key — Aggressive and Conservative run in
        # parallel; both writing to risk_debate_state would collide because
        # langgraph cannot apply a reducer to a nested TypedDict.
        return {"aggressive_risk_argument": argument}

    return aggressive_node
