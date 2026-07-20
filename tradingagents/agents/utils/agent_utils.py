import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.dataflows.symbol_utils import is_cn_share

logger = logging.getLogger(__name__)


def safe_llm_invoke(runnable, messages, label: str = "agent"):
    """Invoke a runnable (LLM or prompt|llm chain) with graceful degradation.

    On any exception (422 content-filter, timeout, rate-limit) logs + returns a
    fallback ``AIMessage`` with empty content + no tool_calls, so the calling
    node degrades (empty report / degraded turn) instead of crashing the whole
    graph. Without this a single provider-side rejection kills the pipeline.
    """
    try:
        return runnable.invoke(messages)
    except Exception as e:  # noqa: BLE001 — must catch all to degrade
        logger.warning(
            "%s invoke failed — degrading (graph continues): %s: %s",
            label, type(e).__name__, str(e)[:200],
        )
        return AIMessage(content=f"({label} analysis unavailable: {type(e).__name__})")


# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.report_digest import build_digest
from tradingagents.agents.utils.web_search_tools import web_search
from tradingagents.agents.utils.technical_indicators_tools import get_indicators

# Public surface: the data tools are imported here so agents and the graph
# import them from one place, plus the instrument/language helpers defined below.
__all__ = [
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_verified_market_snapshot",
    "web_search",
    "build_instrument_context",
    "resolve_instrument_identity",
    "get_instrument_context_from_state",
    "get_language_instruction",
    "format_fair_value_block",
    "collect_analyst_reports",
    "price_snapshot_for",
    "_NO_FABRICATED_BASE_RATES",
    "create_msg_delete",
    "chat_prompt_messages",
    "build_digest_for_state",
]


# Shared pipeline preamble — byte-identical across all 7 analysts.
# Prepended to every analyst's system message so the LLM knows:
#  a) what data was pre-fetched and can be trusted (no recalculation needed)
#  b) who is upstream (already produced reports to read) and downstream (will consume your output)
#  c) common failure modes to avoid (specific to this pipeline, not generic advice)
# The block is pure static text → first agent pays the cache miss; the remaining
# 6 get 100% system-prefix cache hit because the preamble byte range is identical.
_PIPELINE_PREAMBLE = """PIPELINE — TradingAgents 7-analyst chain for the current instrument.

=== PRE-FETCHED DATA (populated before any analyst runs — trust, do NOT recalculate) ===
• PRICE: provided in your DATA block below (the block labeled "PRICE (do NOT modify — sina minute bars)"). It was fetched from sina stock_zh_a_minute, split-adjusted (qfq), real-time during market hours. This IS the current price for this analysis run. Use it as-is for every upside/downside, PE, PB, and yield computation. If you are tempted to re-derive the price from PE/PB triangulation, split data, or book-value adjustments — STOP. The value in your DATA block is the single source of truth.
• FINANCIAL STATEMENTS: fetched from eastmoney stock_financial_abstract. Every number (revenue, net profit, gross margin, debt ratio, EPS, BVPS, ROE, OCF, FCF/share) is SOURCE-COMPUTED by eastmoney — not derived by us. If a cell is blank or NaN in the tool output, report it as unavailable. Do not compute it from nearby rows (e.g. do not derive gross margin from 营业成本 — the source already computes 毛利率 at 18.81% for BYD).
• NEWS BLOCK: pre-fetched once (7-day window, eastmoney headlines) and shared by Sentiment + News analysts. No need to re-fetch get_news for this window.
• MACRO SERIES: fetched on-demand by the Macro & Policy Analyst via get_macro_indicators (LPR/SHIBOR/M2/CPI from akshare). NOT pre-fetched — other analysts should read the macro_policy_report rather than assuming macro data is in their DATA block.

=== ANALYST CHAIN ===
Step 1: Market Analyst — reads price chart; outputs trend, levels, indicator readings.
Step 2: Sentiment Analyst — reads retail chatter (股吧 guba for CN, StockTwits+Reddit otherwise).
Step 3: News Analyst — catalogues events, labels fact/speculation, tags pricing-status per event.
Step 4: Fundamentals Analyst — values the company (PE/PB/DCF/PEG/GQS). Uses the pre-fetched price as-is.
Step 5: Macro & Policy Analyst — quantifies monetary/fiscal/regulatory/real-economy/liquidity backdrop.
Step 6: Business Analyst — breaks down revenue by segment and sub-brand; competitive map; product pipeline.
Step 7: Situation Analyst — synthesises the dated price-journey narrative from all prior reports.

=== DATA TRUST RULES ===
1. Pre-fetched data (price, financials, macro series) is authoritative. Do not second-guess it with training-data priors or PE/PB back-calculation.
2. If a number appears in a tool output, cite it verbatim with the tool name. Do not round or adjust it.
3. If a number is NOT in any tool output, say "not available" — do not estimate from adjacent rows.
4. If your role is to value (Fundamentals), do not also analyse segment structure (Business). If your role is to map events (News), do not also compute fair value. Stay in lane.
5. Common failure mode: Fundamentals rejecting the pre-fetched price and self-calculating 280 from implied PE/PB when the real price is 87. The pre-fetched price IS the current price. Use it.

=== OUTPUT FORMAT RULES ===
- Start your response DIRECTLY with the report content. NO preamble — no "Now I have all data", no "Let me compile", no transition sentences. The first character of your response must be the first section header (e.g. "## 1. Trend" or "# 估值分析").
- After tool-calling finishes, write the final report immediately. The tool results are in your context — do not narrate that you've received them.
"""

# Shared constant — duplicated verbatim across Aggressive/Conservative before Fix 17.
# Use by importing and interpolating into the debator's HARD-CONSTRAINTS block.
_NO_FABRICATED_BASE_RATES = (
    "**NO FABRICATED BASE-RATES (Fix 10/17)**: this node has no tools bound — "
    "you CANNOT pull historical samples, A-share sector stats, or backtests. "
    "Any figure of the form \"n=X 胜率 Y%\" or \"Z 年间同类案例回撤 W%\" MUST "
    "cite the upstream report or web_search result (via fundamentals_report / "
    "news_report / situation_report / business_report digest). If no such "
    "citation exists, the figure is hallucinated and you MUST drop it. "
    "Re-state uncertainty qualitatively (\"low-probability scenario\") rather "
    "than quantifiably (\"5/9 wins\")."
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language and style.

    Returns the language directive (when non-English) appended with a
    concise, "caveman-style" prose instruction. The style directive is
    appended unconditionally because the brevity rule applies even to
    English output — the previous default let analysts ramble to ~5K chars
    per report, which multiplied by the 5×3=15 risk-debate rounds produced
    unreadable 50K-byte output files.

    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debators, research manager, trader, and
    portfolio manager.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")

    style = (
        "\n\nSTYLE — caveman: terse, technical, no filler. "
        "Drop articles (a/an/the), filler (just/really/basically), hedging (may/might/could), "
        "and pleasantries. Fragments OK. Keep technical terms exact. "
        "Cite each figure with its tool source. Avoid restating prior conclusions unless this "
        "round explicitly rebuts them — multi-round debate must add new evidence, not repeat."
    )

    if lang.strip().lower() == "english":
        return style
    return f" Write your entire response in {lang}." + style


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per ticker per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).
    """
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    elif is_cn_share(ticker):
        context += (
            " It is a China A/B-share quoted in **CNY** (T+1 settlement; "
            "main-board ±10% / STAR & ChiNext ±20% daily price limits apply). "
            "Treat every price and fundamental figure as CNY, not USD."
        )
    return context


def format_price_context(state: Mapping[str, Any]) -> str:
    """Render the two pre-fetched price snapshots as a labelled prompt block.

    Distinguishes the **realtime intraday** minute close (pre-settlement,
    moves during the session) from the **settled daily** close (last official
    close, qfq-adjusted). Downstream agents must use the realtime value for
    "current price" / upside-downside and the settled value for historical /
    valuation anchoring. Returns "" when neither is available.

    The header now uses explicit field names (REALTIME/SETTLED) so the
    LLM cannot confuse them — Fix 6.
    """
    rt = state.get("latest_close_realtime")
    rt_at = state.get("latest_close_realtime_at")
    settled = state.get("latest_close")
    settled_at = state.get("latest_close_settled_at")
    if rt is None and settled is None:
        return ""
    lines = [
        "PRICE CONTEXT (do NOT modify — sina minute bars):",
        "  Two distinct price snapshots, each with a SINGLE mandated use:",
        "",
        "  REALTIME = latest intraday minute close (pre-settlement, moves during session)",
    ]
    if rt is not None:
        lines.append(f"    -> {rt} CNY @ {rt_at}")
    lines.append("    USE FOR: 'current price', upside/downside %, gap to target/entry/invalidation.")
    lines.append("")
    lines.append(
        "  SETTLED = last official daily close (qfq-adjusted, post-settlement)"
    )
    if settled is not None:
        lines.append(f"    -> {settled} CNY @ {settled_at}")
    lines.append(
        "    USE FOR: PE / PB / PS / valuation multiples, comparing to historical "
        "earnings/financial-statement periods."
    )
    return "\n".join(lines) + "\n"


def format_fair_value_block(state: Mapping[str, Any]) -> str:
    """Render the deterministic fair_value_block as a labelled prompt block.

    The fair_value_block is pre-computed in propagation.py by
    calculate_fair_value(ticker) — the same function Fundamentals Analyst
    would call as a tool. Injecting the cached block here is the single
    source of truth: Fundamentals, Neutral, Bear, and PM all read this
    instead of each independently reconstructing FV (which produced
    divergent numbers: PM 88 vs Neutral 99 vs Fundamentals 80).

    Returns "" when no block was pre-fetched (caller falls back to tool call
    or to fundamentals_report for FV).
    """
    fv = state.get("fair_value_block")
    if not fv:
        return ""
    return (
        "FAIR VALUE (deterministic 5-model — pre-computed, do NOT recompute "
        "with another tool call; this is the single source of truth):\n"
        "<fair_value>\n"
        f"{fv}\n"
        "</fair_value>\n"
    )


def collect_analyst_reports(state: Mapping[str, Any]) -> str:
    """Concatenate the seven analyst reports in canonical order.

    Fallback used by the researchers/debators when ``report_digest`` is absent
    (bare test states). In the live graph the Report Digest node always
    populates ``report_digest`` first, so this is rarely hit — but keeping the
    fallback in one place stops the 7-way string concat from being copy-pasted
    across every researcher and debator.
    """
    return (
        state.get("market_report", "")
        + state.get("sentiment_report", "")
        + state.get("news_report", "")
        + state.get("fundamentals_report", "")
        + state.get("macro_policy_report", "")
        + state.get("business_report", "")
        + state.get("situation_report", "")
    )


def price_snapshot_for(state: Mapping[str, Any]) -> dict:
    """Return a structured dict of price snapshots for code-side computation.

    Downstream agents that compute PE/PB/upside-downside should call this
    helper instead of re-extracting from state["latest_close_*"] themselves —
    single source of truth for Fix 6.

    Returns:
        {
            "realtime": float | None,
            "realtime_at": str | None,
            "settled": float | None,
            "settled_at": str | None,
            "rule_realtime_for": "current price; upside/downside; gap to target",
            "rule_settled_for": "PE; PB; PS; valuation multiples vs earnings period",
        }
    """
    return {
        "realtime": state.get("latest_close_realtime"),
        "realtime_at": state.get("latest_close_realtime_at"),
        "settled": state.get("latest_close"),
        "settled_at": state.get("latest_close_settled_at"),
        "rule_realtime_for": "current price; upside/downside; gap to target",
        "rule_settled_for": "PE; PB; PS; valuation multiples vs earnings period",
    }


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def build_digest_for_state(state: Mapping[str, Any]) -> str:
    """Build the analyst-report digest from the current graph state.

    Reads the four analyst reports, instrument context, and verified snapshot
    from ``state`` and returns a single bounded digest that downstream
    debate/decision nodes consume instead of reading the full reports.
    Missing fields are tolerated (empty string for absent reports) so this
    helper is safe to call from any partial state during a run.
    """
    return build_digest(
        {
            "market_report": state.get("market_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "news_report": state.get("news_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
            "macro_policy_report": state.get("macro_policy_report", ""),
            "business_report": state.get("business_report", ""),
            "situation_report": state.get("situation_report", ""),
        },
        instrument_context=state.get("instrument_context", ""),
        verified_snapshot=state.get("verified_snapshot"),
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages


# Cache the base ChatPromptTemplate once at module load.
# The system message MUST be pure static text — every agent whose template
# is byte-identical across runs gets 100% cache hit on the system prefix.
# Dynamic content (reports, prices, raw data blocks) goes into the data_block
# user message below, NOT into the system message.
_BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    ("user", "Today's date is {current_date}. Instrument context: {instrument_context}.{data_block}"),
    MessagesPlaceholder(variable_name="messages"),
])


def chat_prompt_messages(system_message, tools, current_date, instrument_context, data_block=""):
    """Compact ChatPromptTemplate for tool-calling analysts.

    Reuses a module-level base template (saving ~100ms per call by avoiding
    the import and construction on every invocation). ``system_message`` must
    be purely static (no f-string, no ``.replace``, no variable injection) so
    the LLM provider caches it across runs. ``data_block`` is where dynamic
    content (analyst reports, price snapshots, raw macro data) lives — it
    follows the static system prefix and changes per run.

    Returns a template with fields pre-bound; the caller does
    ``format_messages(messages=...)`` at invoke time.
    """
    return (
        _BASE_PROMPT.partial(system_message=system_message)
        .partial(current_date=current_date)
        .partial(instrument_context=instrument_context)
        .partial(data_block=data_block)
    )



