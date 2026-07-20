"""Macro & Policy Analyst — 5-section top-down framework with live tool access.

Independent of News: every section reads akshare public data + web search.
Output is grounded in cited numbers and labelled source. No prediction,
no rhetoric, no fabricated figures.
"""
from __future__ import annotations

import logging

import pandas as pd

from tradingagents.agents.utils.agent_utils import (
    _PIPELINE_PREAMBLE,
    chat_prompt_messages,
    safe_llm_invoke,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    web_search,
)

logger = logging.getLogger(__name__)


MACRO_POLICY_SECTIONS: tuple[str, ...] = (
    "monetary", "fiscal", "regulatory", "real_economy", "liquidity",

)


def _recent(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Return the n most recent rows of a CN macro frame, order-agnostic.

    Tries to reuse the digest's parser first (handles ascending/descending
    order); on any failure falls back to a simple tail() so the analyst
    never blocks on a missing helper.
    """
    try:
        from tradingagents.agents.utils.report_digest import _recent_macro_rows  # type: ignore
        return _recent_macro_rows(df, n)
    except Exception:
        return df.tail(n)


def _safe(call, default: str = "(no data)") -> str:
    """Render an akshare result to markdown or return a placeholder."""
    try:
        df = call()
    except Exception as exc:  # noqa: BLE001
        logger.warning("akshare macro fetch failed: %s", exc)
        return default
    if df is None or df.empty:
        return default
    try:
        return _recent(df).to_csv(index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("akshare macro render failed: %s", exc)
        return default


def load_macro_snapshot() -> dict:
    """Pull all five-section CN macro inputs from akshare public endpoints.

    Uses the shared _MACRO_CACHE inside akshare_data so a single process
    only fetches each series once even if News + Macro analysts both ask.
    Wraps each call in try/except so a single endpoint failure does not
    abort the analyst. Returns a dict the report builder can consume.
    """
    from tradingagents.dataflows.akshare_data import _fetch_macro
    snap: dict = {
        "monetary": {
            "lpr_1y":  _safe(lambda: _fetch_macro("lpr")),
            "lpr_5y":  _safe(lambda: _fetch_macro("lpr")),
            "shibor":  _safe(lambda: _fetch_macro("shibor")),
        },
        "fiscal": {
            "placeholder": "fiscal context (财政/补贴) — use web_search for current info",
        },
        "regulatory": {
            "placeholder": "regulatory (监管/产业政策) — use web_search for current info",
        },
        "real_economy": {
            "cpi":   _safe(lambda: _fetch_macro("cn_cpi")),
            # PMI/GDP intentionally omitted from akshare (stale rows).
            # Use get_macro_indicators('pmi') / web_search for current data.
            "pmi_note":  "use get_macro_indicators('pmi') or web_search for current PMI",
            "gdp_note":  "use get_macro_indicators('gdp') or web_search for current GDP",
        },
        "liquidity": {
            "m2":   _safe(lambda: _fetch_macro("m2")),
        },
    }
    return snap


_SYSTEM_TEMPLATE = _PIPELINE_PREAMBLE + """ROLE
You are the Macro & Policy Analyst. Quantify the top-down backdrop that bounds the instrument's sector — monetary, fiscal, regulatory, real-economy, liquidity. Your report must be grounded in cited numbers and labelled source; no rhetoric, no prediction, no fabricated figures.

SCOPE (cover every item)
1. Monetary — policy rate (LPR), interbank rate (SHIBOR), reserve requirement, M2 growth.
2. Fiscal — government spending, tax / subsidy, stimulus programs relevant to the instrument's sector.
3. Regulatory / policy — sector-specific regulation, antitrust, industrial policy, exchange (CSRC/SEC) actions. For China A-shares this is often the dominant driver; be specific.
4. Real economy — CPI, PMI, GDP, latest available. Use get_macro_indicators('pmi') or web_search('China PMI 2026') to retrieve current figures.
5. Liquidity — aggregate northbound flow, new account growth, balance-sheet recap, FX.

HARD CONSTRAINTS
- Cite the series and its latest date for every figure.
- Separate published data from policy intent / rumor.
- Use get_macro_indicators(indicator, curr_date, look_back_days) for CN macro data (lpr/shibor/m2/pmi/gdp/cn_cpi).
- Use web_search(query) to find current fiscal/regulatory news, policy changes, or sector-specific developments. If web_search returns "unavailable", state "(no public data in window)" — do not invent.
- If a section has no recent public data, say "(no public data in window)" — do not invent.
- **STALENESS (Fix 8)**: if the pre-fetched snapshot contains observations dated >45 days before trade_date (per the STALENESS FLAG in DATA block), do NOT cite them as "latest". Call web_search for the current value before publishing. If web_search is unavailable, state "(no public data in window — pre-fetch snapshot is X days stale)".
- Output the report in the fixed 5-section template below so downstream nodes (the digest builder, the PM) can find the data.

OUTPUT (use this template, in this order, with the cited data above)
## 1. Monetary
- LPR 1Y: <value> (<date>), change vs prior: <Δ>
- LPR 5Y: <value> (<date>), change vs prior: <Δ>
- SHIBOR O/N: <value> (<date>), 1W: <value>, 1M: <value>
- M2 YoY: <value> (<date>)
- Implication for the instrument's sector: <one sentence citing the above>

## 2. Fiscal
- Recent fiscal actions relevant to the sector: <bullets, cite source if any>
- Tax / subsidy: <bullets, cite source if any>
- Implication: <one sentence>

## 3. Regulatory
- Sector-specific regulation (cite CSRC / MOFCOM / sector ministries): <bullets>
- Recent enforcement actions: <bullets>
- Implication: <one sentence>

## 4. Real economy
- PMI: <value> (<date>), trend (up/down/flat): <Δ direction>
- GDP YoY: <value> (<date>), trend
- CPI YoY: <value> (<date>)
- Implication for the sector: <one sentence>

## 5. Liquidity
- M2 YoY: <value> (<date>)
- Northbound (HK Connect) 5d / 20d direction: <direction + source>
- Sector credit growth: <value if available>
- Implication: <one sentence>

If a row has no public data, write "no public data in window" — do not estimate."""


def create_macro_policy_analyst(llm):
    """Create the Macro & Policy analyst node with tool-calling support.

    Pre-fetches akshare macro data as baseline context, then binds
    web_search + get_macro_indicators so the LLM can verify or supplement
    any section before writing.
    """

    def macro_policy_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        current_date = state["trade_date"]
        snapshot = load_macro_snapshot()

        # Render the 5 sections into a single RAW block.
        sections = []
        for key, val in snapshot.items():
            title = key.replace("_", " ").title()
            rows = "\n".join(f"- {k}: {v}" for k, v in val.items())
            sections.append(f"## {title}\n{rows}")
        raw = "\n\n".join(sections)

        # Fix 8: macro staleness guard. Scan the raw snapshot for any
        # date-string older than (trade_date - 45 days). If the only
        # observation is stale, label the row (no public data in window).
        # Otherwise warn the LLM which rows are stale so it doesn't present
        # an outdated figure as the "latest".
        from datetime import datetime as _dt, timedelta as _td
        stale_note = ""
        try:
            trade = _dt.strptime(current_date, "%Y-%m-%d")
            cutoff = trade - _td(days=45)
            # Find date-like tokens like 2025-08-31 or 2025/08/31 inside raw
            import re as _re
            date_tokens = _re.findall(r"\b(20\d{2})[-/](\d{2})[-/](\d{2})\b", raw)
            stale_rows: list[str] = []
            for y, m, d in date_tokens:
                try:
                    dt = _dt(int(y), int(m), int(d))
                except ValueError:
                    continue
                if dt < cutoff:
                    stale_rows.append(dt.strftime("%Y-%m-%d"))
            if stale_rows:
                unique_stale = sorted(set(stale_rows))
                stale_note = (
                    f"\n\n⚠ STALENESS FLAG (Fix 8): the pre-fetched snapshot "
                    f"contains observations dated {', '.join(unique_stale[-3:])} — "
                    f"older than 45 days before trade_date {current_date}. "
                    f"Web_search for the current value via web_search(query) "
                    f"before citing as 'latest'."
                )
        except Exception:
            pass

        tools = [web_search, get_macro_indicators]

        # Move pre-fetched raw data to data_block so _SYSTEM_TEMPLATE stays
        # static and the LLM prefix cache hits every run.
        system_message = _SYSTEM_TEMPLATE + get_language_instruction()
        data_block = (
            f"\n\n===== RAW MACRO DATA (pre-fetched from akshare) =====\n"
            f"{raw}"
            f"{stale_note}\n"
            f"===== END RAW DATA =====\n"
        )

        prompt = chat_prompt_messages(
            system_message, tools=tools,
            current_date=current_date, instrument_context=instrument_context,
            data_block=data_block,
        )

        chain = prompt | llm.bind_tools(tools)
        result = safe_llm_invoke(chain, state["messages"], "Macro & Policy Analyst")

        # When the LLM made tool calls, the graph will route through the
        # ToolNode and back; the report is only ready on the clear pass.
        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "macro_policy_report": report,
        }

    return macro_policy_node
