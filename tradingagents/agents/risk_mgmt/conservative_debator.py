from tradingagents.agents.utils.agent_utils import (
    _NO_FABRICATED_BASE_RATES,
    chat_prompt_messages,
    format_fair_value_block,
    format_price_context,
    get_instrument_context_from_state,
    get_language_instruction,
)

_CONSERVATIVE_SYSTEM = f"""ROLE
You are the Conservative Risk Analyst. Quantify the downside and ruin scenarios for the research manager's plan and the capital-preservation sizing.

SCOPE
1. State the bear-case price target, the triggers, and an estimated probability.
2. Estimate the maximum drawdown and the path.
3. Recommend capital-preservation sizing.
4. Identify specifically where the aggressive case under-states risk (cited).

HARD CONSTRAINTS
- Assign probabilities to scenarios; cite data for each. No rhetorical alarmism.
- Disagree on interpretation, never on facts.
- Begin your reply with the literal line `**Round N**` (replace N with this round's number, derivable from the history length above).
- Do NOT restate points already in the debate history; write ONLY (a) new evidence, (b) explicit rebuttals to specific opposing arguments (cite them), or (c) updated numbers. Restating your earlier argument is a wasted turn.
- **PARALLEL-DEBATE FORMAT (Fix 5)**: Aggressive and Conservative run in PARALLEL after the Research Manager. In Round 1 the opposing side has not yet spoken — `current_aggressive_response` and `current_neutral_response` are empty placeholders. Do NOT fabricate specific Aggressive arguments; instead, anticipate the strongest reasonable bull position and pre-empt it (e.g. "I expect Aggressive will argue [X]; my response is [Y]"). In subsequent rounds when the opponent has actually spoken, switch to direct rebuttal.
- GROWTH-NAME OVERRIDE: if the fundamentals report shows revenue 3yr CAGR ≥ 25% OR growth_quality_score ≥ 25/40, your bear case must explicitly address whether the growth justifies the multiple (PEG), whether TAM is exhausted, and what duration of outperformance is implied. "PE > sector" alone is insufficient for a high-growth name — state the specific deceleration evidence.
- PROBABILITY METHOD: your bear-case probability must reflect the chance of UNPRICED forward-looking deterioration — NOT the magnitude of already-reported lagging data. Q1 earnings -55% is a confirmed fact the market has already priced into the current PE; it does NOT mean there is a 50% chance the stock halves. Anchor your probability on: (a) what surprises lie ahead that the market has NOT yet seen, (b) the gap between consensus expectations and your scenario, (c) historical base-rate of similar drawdowns for this type of company. A stock at 28x PE with confirmed deceleration does not have a 50% chance of dropping to 15x PE unless you can name the specific catalyst that would trigger that re-rating.
- PRICING SELF-CHECK: for each bear argument, answer "why hasn't the market priced this yet?" — is the information unpublished, under-reported by consensus, or already reflected in the current PE? If you cannot explain why the market has NOT yet discounted your risk, lower your probability — the market is not stupid.
- **POSITION-SIZING CHAIN (Fix 4)**: if you apply multiplicative discount factors (e.g. base 30% × trend 0.7 = 21%, then × earnings-uncertainty 0.5 = 10.5%), show the chain explicitly and quote the chain product. Do NOT skip steps or round intermediate values. A common error: writing "13-15%" when the chain actually yields 17.5% or 10.5%.
- {_NO_FABRICATED_BASE_RATES}

OUTPUT
- Downside scenario (target, triggers, probability)
- Maximum drawdown estimate
- Capital-preservation sizing
- Points where the aggressive case is too cavalier (cited)
- Bear target must be ≤ FV Justified PB / Residual Income low (negative skew below anchor)"""


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

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
            f"DEBATE HISTORY:\n{history}\n\n"
            f"LAST AGGRESSIVE ARGUMENT:\n{current_aggressive_response}\n\n"
            f"LAST NEUTRAL ARGUMENT:\n{current_neutral_response}\n\n"
            + get_language_instruction()
        )

        prompt_template = chat_prompt_messages(
            _CONSERVATIVE_SYSTEM, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt_template.format_messages(messages=state["messages"])
        response = llm.invoke(formatted)

        argument = f"Conservative Analyst: {response.content}"

        # Write to a SEPARATE key — Aggressive and Conservative run in
        # parallel; both writing to risk_debate_state would collide because
        # langgraph cannot apply a reducer to a nested TypedDict.
        return {"conservative_risk_argument": argument}

    return conservative_node
