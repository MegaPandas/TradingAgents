"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import re

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    chat_prompt_messages,
    format_price_context,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.symbol_utils import is_cn_share

logger = __import__("logging").getLogger(__name__)


def _enforce_hold_if_fv_within_tolerance(
    decision_text: str,
    fv_block: str,
    current_price: float | None,
    tolerance_pct: float = 5.0,
) -> str:
    """Post-hoc check: if FV is within tolerance of price, force Hold rating.

    The PM prompt says \"fair value ~= price (+/-5%) -> Hold\" but the LLM
    sometimes overrides this when the free-text path bypasses the structured
    validator.  This function regex-extracts the PM's own adjusted FV (or
    falls back to the normalized median from the fair_value block), computes
    the gap, and patches the decision text if the PM's rating contradicts
    the rule.

    Returns the (possibly patched) decision text.
    """
    if current_price is None or current_price <= 0:
        return decision_text

    # 1. Extract the PM's claimed adjusted FV from the decision text.
    #    Patterns: "FV (adjusted) | 90.18", "adjusted FV = 90.18",
    #    "调整后 = 85.08 × 1.06 = 90.18", "fair value ... 90.18"
    fv_patterns = [
        # Table row: "FV (adjusted) | 90.18" or "| FV (adjusted) | 90.18 |"
        r"(?:FV\s*\(adjusted\)|adjusted?\s*(?:FV|fair|公允))[^\n]*?\|?\s*([-+]?\d+\.?\d{1,2})\b",
        # Math chain: "85.08 × 1.06 = 90.18"
        r"(?:85\.0?8?\d*\s*[×x]\s*\d+\.\d+\s*=\s*)([-+]?\d+\.?\d{1,3})\b",
        # English prose: "adjusted FV ≈ 90.18" / "adjusted to 90.18"
        r"(?:adjusted?\s*(?:FV|to)\s*(?:≈|~|=|is)?\s*)([-+]?\d+\.?\d{1,2})\b",
        # Chinese prose: "调整后 (FV / 公允价值) ... 90.18"
        r"(?:调整后|归一化调整后).*?([-+]?\d+\.?\d{1,2})\s*(?:CNY|💰|元)?",
        # "目标 91.46" / "加权目标 91.46" (when no adjusted FV found)
        r"(?:加权)?目标\s*(?:价|值)?.*?([-+]?\d+\.?\d{1,2})\s*(?:CNY|💰|元)?",
    ]
    pm_fv: float | None = None
    for pat in fv_patterns:
        m = re.search(pat, decision_text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                # Sanity: reject unreasonable values (< 1 or > 10000)
                if 1.0 < val < 10000.0:
                    pm_fv = val
                    break
            except ValueError:
                continue

    # 2. Fallback: use normalized median from fair_value_block
    if pm_fv is None and fv_block:
        for pat in [
            # "- **Median**: 85.08" after "Normalized-EPS"
            r"Normalized-EPS.*?[-*]*\s*\**Median\**[:\s]*([-+]?\d+\.?\d{1,2})",
            # "Normalized-EPS Fair Value" section with median
            r"Normalized.*?Fair\s+Value.*?median[:\s]*([-+]?\d+\.?\d{1,2})",
            # Generic median line
            r"\*\*Median\*\*[:\s]*([-+]?\d+\.?\d{1,2})",
        ]:
            norm_match = re.search(pat, fv_block, re.DOTALL | re.IGNORECASE)
            if norm_match:
                try:
                    val = float(norm_match.group(1))
                    if 1.0 < val < 10000.0:
                        pm_fv = val
                        break
                except ValueError:
                    continue

    if pm_fv is None:
        return decision_text

    # 3. Compute gap
    gap_pct = abs(pm_fv - current_price) / current_price * 100.0
    if gap_pct > tolerance_pct:
        return decision_text

    # 4. Extract rating
    rating_match = re.search(
        r"\*\*Rating\*\*[:\s]*(Buy|Overweight|Hold|Underweight|Sell)",
        decision_text, re.IGNORECASE,
    )
    if not rating_match:
        return decision_text

    current_rating = rating_match.group(1)
    if current_rating == "Hold":
        return decision_text

    # 5. Patch: replace the first occurrence of the non-Hold rating
    logger.warning(
        "Portfolio Manager: FV %.2f vs price %.2f = %.1f%% gap (<= %.0f%% "
        "Hold threshold), but PM chose %s. Forcing Hold per HARD CONSTRAINTS.",
        pm_fv, current_price, gap_pct, tolerance_pct, current_rating,
    )

    # Patch the Rating line
    decision_text = decision_text.replace(
        f"**Rating**: {current_rating}",
        f"**Rating**: Hold  [forced: FV {pm_fv:.2f} vs price {current_price:.2f} = {gap_pct:.1f}% <= {tolerance_pct:.0f}% Hold threshold]",
    )
    # Also try Chinese variant
    decision_text = re.sub(
        r"(评级[：:]\s*)" + current_rating,
        rf"\1Hold  [forced: FV {pm_fv:.2f} vs price {current_price:.2f} = {gap_pct:.1f}% <= {tolerance_pct:.0f}% Hold threshold]",
        decision_text,
    )
    return decision_text

_PM_SYSTEM = """ROLE
You are the Portfolio Manager. Synthesize the risk analysts' debate and deliver the final decision with a complete, executable trade plan.

DECISION PROCESS (follow this order — each step gates the next)
1. FAIR VALUE: the `<fair_value>` block contains TWO ranges when the tool detected a cyclical trough (TTM EPS < 85% of FY EPS):
   - **TTM-anchor range** — uses depressed TTM EPS (e.g. Q1 profit -55%). Models will understate.
   - **Normalized-EPS range** — uses FY EPS as the cyclical-anchor input. Closer to fair economic value.
   Read BOTH medians. Use the **normalized-EPS median as your PRIMARY anchor** when the tool flagged a cyclical trough (the warning is printed in the block). Adjust only for factors the deterministic models can't see: structural mix shift (export share rising), technology/regulatory moat, expected ROE trajectory. Show your adjusted FV AND the unadjusted normalized median side by side. List every adjustment as an explicit +/- %; verify with `validate_fair_value_adjustment(anchor, adjustments_pct, adjusted_FV)` mentally — the product must equal adjusted FV within 1%. If your printed deltas don't reconcile, re-compute before publishing. Do NOT derive FV from debate price targets (120/85/68) — those are scenario endpoints, not fair value inputs.
2. RATING: compare fair value vs current price. If fair value >> price -> Buy/Overweight. If fair value ~= price (+/-5%) -> Hold. If fair value << price -> Underweight/Sell.
3. TRADE PLAN: derive target/entry/invalidation from fair value and technical levels. Target = your fair value (or a scenario-adjusted version). Entry = a pullback level below current price for Buy, or below fair value for Underweight reload. Invalidation = the level that breaks your thesis.
4. TIME HORIZON.
5. If your final rating differs from the Research Manager's, state why in one sentence.

HARD CONSTRAINTS
- Every price level must trace to a cited number from the analyst reports. Do not invent round numbers or levels that do not appear in the source analysis.
- The price_target must name the method and the EPS period it rests on (e.g. "25x PE on 2025 EPS 3.577 = 89.43"). A bare float with no derivation method is not acceptable.
- INTERNAL CONSISTENCY: rating, price_target, and entry_price must point the same direction:
  * Buy/Overweight -> price_target > current price; entry_price <= current price (buy on weakness).
  * Hold -> price_target ~= current price.
  * Underweight -> price_target <= current price (your fair value is BELOW current); entry_price < price_target (reload on deeper pullback).
  * Sell -> price_target < current price; no entry (thesis broken).
  Do NOT mix: an Underweight rating with a bull-case target above the current price is a contradiction. If your fair value is above the current price, the rating is not Underweight.
- The trade plan is required whether the rating is bullish or bearish.
- Ground every conclusion in specific evidence from the analysts.

HARD CONSTRAINT (cross-citation)
- The investment_thesis must name at least one specific point from EACH
  side of the risk debate (Aggressive + Conservative). Do not pick the
  probability-weighted number and then ignore the side that lost the
  arithmetic. Cite the strongest claim that did NOT win and explain
  why the other side carried the verdict anyway.
- Example: "Aggressive cited PEG=1.25 as fair-value support; Conservative
  countered that FY25 profit -19% means revenue growth isn't converting
  to earnings. The probability weight leans bearish because..."

OUTPUT
Structure your decision as follows:
- **Rating**: one of Buy / Overweight / Hold / Underweight / Sell
- **Decision Card** (one compact block — the reader's first stop):
  | Field | Value |
  |---|---|
  | Rating | <rating> |
  | Current price | <from data block> |
  | FV (TTM anchor) | <5-model median using depressed TTM EPS — only useful as downside sanity check> |
  | FV (normalized anchor) | <5-model median using FY EPS — PRIMARY anchor when cyclical trough detected> |
  | FV (adjusted) | <your adjusted value> (<adjustment: e.g. export-share 39%->45%+ on normalized EPS -> +12%>) |
  | Target price | <value> (<method: e.g. 25x PE on normalized FY2025 EPS 3.58 = 89.5>) |
  | Entry price | <value> (<source>) |
  | Invalidation | <value> (<source>) |
  | Position sizing | <value> (<basis>) |
  | Time horizon | <value> |
- **Executive Summary**: 2-3 sentences max. State the verdict + the single strongest reason.
- **Investment Thesis**: cite at least one specific point from EACH side of the risk debate.
- **Operation Rules**: numbered if-then list. Each rule: trigger -> action. No prose paragraphs.
- **Notes**: rating_change_rationale (if your rating differs from RM). Keep to 1 sentence."""


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        price_ctx = format_price_context(state)
        price_line = price_ctx.rstrip() if price_ctx else (
            "- PRICE: UNAVAILABLE"
        )
        digest_line = (
            f"- report_digest: present ({len(state['report_digest'])} chars)"
            if state.get("report_digest") else "- report_digest: NOT PRESENT"
        )
        fv_block = state.get("fair_value_block", "")
        if fv_block:
            fv_line = (
                "- fair_value (deterministic 5-model — your PRIMARY anchor):\n"
                f"<fair_value>\n{fv_block}\n</fair_value>"
            )
        else:
            fv_line = "- fair_value: NOT PRE-COMPUTED"

        ticker = state.get("company_of_interest", "")
        cn = is_cn_share(ticker)
        cn_rules = (
            "\n\nMARKET-STRUCTURE OVERRIDE — A-share long-only\n"
            "This is a China A-share. Retail flow is long-only; shorting needs "
            "a margin account with borrowable shares, which most retail desks "
            "do not have. Pick the rating to MATCH the trade plan:\n"
            "- **Buy / Overweight** -> bullish: enter on weakness toward "
            "`entry_price`; the operation_rules describe building a long.\n"
            "- **Hold** -> already long and want to keep it, OR fair value close "
            "to current price with no edge either way; operation_rules describe "
            "holding the existing position.\n"
            "- **Underweight** -> TRIM AND WAIT: trim existing long and "
            "re-enter on a pullback to `entry_price`. The trade plan is a "
            "two-phase exit-then-reload. Do NOT ask the user to short.\n"
            "- **Sell** -> THESIS BREAK: liquidate any existing long AND do NOT "
            "re-enter within the time horizon.\n"
            "For every rating, `entry_price` is the price at which to be LONG. "
            "`invalidation` is the price below which the long thesis breaks. "
            "`price_target` is the take-profit level for the long."
            if cn
            else ""
        )
        growth_overrides = (
            "\n\nGROWTH-NAME ADJUSTMENT\n"
            "When fundamentals flagged this as a growth name — revenue 3yr "
            "CAGR >= 25% OR growth_quality_score >= 25/40 ('quality growth'):\n"
            "- A bare 'PE > sector' bear case is INSUFFICIENT. The bear must "
            "argue growth deceleration, TAM exhaustion, or margin compression "
            "with specific cited evidence, not just multiple compression.\n"
            "- Use growth-adjusted valuation (PEG / growth-stage DCF) as the "
            "primary fair-value anchor, not PE.\n"
            "- For Buy ratings, name the duration of growth runway (e.g. '3+ "
            "years of >25% revenue CAGR visible') that justifies the entry.\n"
            "- Down-weight Underweight/Sell calls when growth_quality_score >= "
            "25 unless deceleration is concrete (specific metric falling for "
            "2+ quarters).\n"
            "\n"
            "NEAR-GROWTH ADJUSTMENT (revenue CAGR 22-24% OR GQS 20-24):\n"
            "- The name is on the boundary. Apply the growth-adjusted valuation "
            "rules above as the primary anchor, but you may still produce "
            "Underweight / Sell ratings if the cited evidence is concrete "
            "(e.g. specific deceleration metric falling for 2+ quarters).\n"
            "- State the boundary status explicitly in the thesis ('on the "
            "boundary between growth and cyclical; verdict depends on whether "
            "revenue growth sustains through [cited quarter]').\n"
            "- Reserve Sell for thesis-break cases only."
        )

        data_block = (
            f"{price_line}\n{digest_line}\n{fv_line}\n\n"
            f"===== MARKET-STRUCTURE RULES =====\n"
            f"{cn_rules if cn else 'Standard (no CN override)'}\n\n"
            f"===== GROWTH ADJUSTMENTS =====\n{growth_overrides}\n\n"
            f"===== DEBATE CONTEXT =====\n"
            f"Research Manager's investment plan: **{research_plan}**\n"
            f"{lessons_line}Risk Analysts Debate History:\n{history}\n\n"
            + get_language_instruction()
        )

        prompt = chat_prompt_messages(
            _PM_SYSTEM, tools=[],
            current_date="", instrument_context=instrument_context,
            data_block=data_block,
        )
        formatted = prompt.format_messages(messages=state["messages"])

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted,
            render_pm_decision,
            "Portfolio Manager",
        )

        # Post-hoc: force Hold if FV is within ±5% of current price.
        # The PM prompt has this rule but the LLM sometimes overrides,
        # especially on free-text fallback when the structured validator
        # is bypassed.
        current_price = state.get("latest_close_realtime") or state.get("latest_close")
        final_trade_decision = _enforce_hold_if_fv_within_tolerance(
            final_trade_decision,
            state.get("fair_value_block", ""),
            current_price,
            tolerance_pct=5.0,
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
