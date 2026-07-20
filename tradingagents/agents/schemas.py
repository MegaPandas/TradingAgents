"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import re

from pydantic import BaseModel, Field, field_validator, model_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Coerce LLM price strings into float | None.

    Models often emit decorated prices like ``"85.08 CNY（25× PE on FY EPS）"``
    even though the field type is ``float``.  Pydantic rejects those.  This
    validator, run ``mode="before"``, extracts the leading numeric token so
    the structured-output call succeeds instead of falling back to free text.
    """
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in _NULLISH_FLOAT:
            return None
        # Extract the first signed decimal number anywhere in the string.
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description=(
            "Target price in the instrument's quote currency. Provide whenever a "
            "position is intended, bullish or bearish (for a bearish call this is "
            "the downside target)."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Actionable entry price in the quote currency. For a bullish rating "
            "the buy level. For A-shares this is ALWAYS a buy-to-enter price, "
            "even under a Sell/Underweight rating (the bear case becomes 'wait "
            "for a pullback to this level'). For shortable markets this can also "
            "be the short entry. Provide whenever a position is intended."
        ),
    )
    invalidation: float | None = Field(
        default=None,
        description=(
            "Price at which the trade thesis is invalidated (in quote currency). "
            "Provide whenever a position is intended, for either direction."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description="Position size guidance, e.g. '3-5% of portfolio'.",
    )
    operation_rules: str = Field(
        description=(
            "Complete, always-present operation plan. Regardless of whether the "
            "rating is bullish or bearish, spell out concrete if-then rules: the "
            "entry trigger, levels at which to add or trim, the exit / "
            "invalidation condition, and the condition that would re-open the "
            "position. Ground every price level in the analyst reports — do not "
            "invent round numbers."
        ),
    )
    rating_change_rationale: str | None = Field(
        default=None,
        description=(
            "If the final rating differs from the Research Manager's or the Trader's, "
            "state why in one sentence."
        ),
    )
    level_delta: str | None = Field(
        default=None,
        description="Summarize adjustments vs the Research Manager's plan (if any).",
    )
    fair_value: float | None = Field(
        default=None,
        description=(
            "Your probability-weighted fair value. Must be computed from "
            "the debate scenarios (weighted average), NOT copied from one "
            "side. Show the weighting formula."
        ),
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", "entry_price", "invalidation", "fair_value", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @model_validator(mode="after")
    def _check_rating_target_consistency(self):
        """Reject direction-mismatched rating + price_target.

        Buy/Overweight → target should be above entry.
        Underweight/Sell → target should be at or below current price context.
        Hold → target ≈ current price.
        This catches the 'Underweight + target 126' contradiction.
        """
        if self.price_target is not None and self.entry_price is not None:
            if self.rating in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT):
                if self.price_target < self.entry_price:
                    raise ValueError(
                        f"INTERNAL CONSISTENCY: {self.rating.value} rating but "
                        f"price_target ({self.price_target}) < entry_price "
                        f"({self.entry_price}). Bullish rating requires target > entry."
                    )
            if self.rating == PortfolioRating.UNDERWEIGHT:
                if self.price_target > self.entry_price * 1.3:
                    raise ValueError(
                        f"INTERNAL CONSISTENCY: Underweight rating but price_target "
                        f"({self.price_target}) >> entry_price ({self.entry_price}). "
                        f"Underweight means fair value is BELOW current — target "
                        f"should not be a bull-case number."
                    )
        return self


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    plan = []
    if decision.price_target is not None:
        plan.append(f"- Target price: {decision.price_target}")
    if decision.entry_price is not None:
        plan.append(f"- Entry: {decision.entry_price}")
    if decision.invalidation is not None:
        plan.append(f"- Invalidation: {decision.invalidation}")
    if decision.position_sizing:
        plan.append(f"- Position sizing: {decision.position_sizing}")
    if plan:
        parts.extend(["", "**Trade Plan**", "\n".join(plan)])
    parts.extend(["", "**Operation Rules**", decision.operation_rules])
    if decision.rating_change_rationale:
        parts.extend(["", "**Rating Change Rationale**", decision.rating_change_rationale])
    if decision.level_delta:
        parts.extend(["", "**Level Delta vs Trader**", decision.level_delta])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex. Labels follow the configured output language so a
    Chinese-mode run does not produce an English header mixed into an
    otherwise-Chinese report (the FORMAT RULE now anchors every analyst to
    Chinese from its first character, and this header follows suit).
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    cn = lang.strip().lower() != "english"
    if cn:
        band_map = {
            "Bullish": "看多", "Mildly Bullish": "轻度看多",
            "Neutral": "中性", "Mixed": "分歧",
            "Mildly Bearish": "轻度看空", "Bearish": "看空",
        }
        return "\n".join([
            f"**整体情绪:** **{band_map.get(report.overall_band.value, report.overall_band.value)}** "
            f"(评分: {report.overall_score:.1f}/10)",
            f"**置信度:** {report.confidence.capitalize()}",
            "",
            report.narrative,
        ])
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
