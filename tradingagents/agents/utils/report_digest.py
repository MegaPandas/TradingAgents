"""Deterministic analyst-report digest — single source of truth for downstream.

Each analyst's report is condensed into a 'Key findings' section that retains
the first heading + the leading prose of the body. Numbers and dates flow
naturally with the prose (no separate list), so downstream agents can read
each analyst's actual reasoning — policy citations, thesis sentences, source
attributions — instead of a compressed list of digits. The total output is
bounded by MAX_LEN so a downstream agent cannot accidentally swallow a multi-KB
blob.
"""
from __future__ import annotations

import re

MAX_LEN = 12000           # ~12 KB — fits 7 analyst sections + snapshot
_PER_SECTION_CAP = 1200   # bytes per analyst section — 7×1200 = 8400 < 12000
_HEADING_RE = re.compile(r"(?m)^#{1,3}\s+.+$")
_SECTION_HEADERS = (
    ("MARKET", "market_report"),
    ("SENTIMENT", "sentiment_report"),
    ("NEWS", "news_report"),
    ("FUNDAMENTALS", "fundamentals_report"),
    ("MACRO & POLICY", "macro_policy_report"),
    ("SITUATION", "situation_report"),
    ("BUSINESS", "business_report"),
)


def _cap(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _key_findings(report: str) -> str:
    """Return the first heading + leading _PER_SECTION_CAP bytes of body.

    The body is preserved (not just numbers/dates) so downstream agents can
    read each analyst's actual reasoning — policy citations, thesis sentences,
    source attributions — not just a list of digits. The heading is repeated
    at the top of the section so a downstream prompt that quotes the digest
    can reference each analyst by section name without re-reading state.
    """
    if not report:
        return "(no report)"
    text = report.replace("<no guba posts found", "(no source)").replace(
        "<stocktwits: not applicable", "(skipped CN)").replace(
        "<reddit: not applicable", "(skipped CN)").replace(
        "<guba: not a China", "(skipped non-CN)")
    heading = next(iter(_HEADING_RE.findall(text)), "")
    body = text
    if heading:
        body = body.replace(heading, "", 1).lstrip("\n")
    body = _cap(body, _PER_SECTION_CAP)
    if heading:
        return heading + "\n" + body
    return body


def build_digest(reports: dict, *, instrument_context: str, verified_snapshot: str | None) -> str:
    """Return a deterministic digest of the 4 analyst reports + snapshot.

    `reports` must contain the keys market_report/sentiment_report/news_report/
    fundamentals_report. Each section is capped so the total output stays
    under MAX_LEN bytes.
    """
    parts = [f"INSTRUMENT: {instrument_context.strip()}"]
    if verified_snapshot and verified_snapshot.strip():
        snap = _cap(verified_snapshot.strip(), _PER_SECTION_CAP)
        parts.append(f"VERIFIED SNAPSHOT:\n{snap}")
    for header, key in _SECTION_HEADERS:
        findings = _key_findings(reports.get(key, ""))
        parts.append(f"{header}: {_cap(findings, _PER_SECTION_CAP)}")
    out = "\n\n".join(parts)
    out = _cap(out, MAX_LEN)

    # Fix 7: cross-report sign-consistency check. If two reports quote
    # numerical figures with opposite signs near a known metric
    # (e.g. H1 cumulative sales), flag so downstream debate is anchored
    # against the conflict instead of citing whichever happened to load.
    conflict_flag = _cross_report_sign_conflict(reports)
    if conflict_flag:
        out += f"\n\nCROSS-REPORT CONFLICT FLAG (Fix 7):\n{conflict_flag}"
    return _cap(out, MAX_LEN + 600)


# Recognised metrics whose +/- direction can conflict across time-windows
# (June single-month vs H1 cumulative).  Add more as they surface.
_METRIC_PATTERNS: dict[str, list[str]] = {
    "sales_yoy_pct": [
        r"6月.*?销量.*?([+-]?\d+\.\d+)\s*%",
        r"六月.*?销量.*?([+-]?\d+\.\d+)\s*%",
        r"H1累计销量.*?([+-]?\d+\.\d+)\s*%",
        r"上半年累计销量.*?([+-]?\d+\.\d+)\s*%",
    ],
    "net_profit_yoy_pct": [
        r"归母净利润.*?同比\s*([+-]?\d+\.\d+)\s*%",
        r"净利同比.*?([+-]?\d+\.\d+)\s*%",
    ],
}


def _cross_report_sign_conflict(reports: dict) -> str:
    """Detect same-metric figures with opposite signs across reports.

    The audit (002594.SZ_20260717_121508) found News citing H1 cumulative
    sales -15.72% while Business cited June single-month +5.46% — both
    correct under different time windows, but downstream agents read one
    or the other and built contradictory bear pillars.  This function
    scans every report for numeric figures near known metrics and flags
    when two signs disagree, even if windows differ (still a hazard:
    downstream debators quote one and ignore the other).
    """
    import re as _re
    findings: list[str] = []
    for metric_name, patterns in _METRIC_PATTERNS.items():
        seen: list[tuple[str, str, float, str]] = []
        for key, report in reports.items():
            if not report:
                continue
            for pat in patterns:
                for m in _re.finditer(pat, report):
                    try:
                        v = float(m.group(1))
                    except (ValueError, IndexError):
                        continue
                    start = max(0, m.start() - 30)
                    end = min(len(report), m.end() + 40)
                    ctx = report[start:end].replace("\n", " ")
                    if "6月" in ctx or "六月" in ctx:
                        period = "monthly"
                    elif "Q1" in ctx or "Q2" in ctx or "Q3" in ctx or "Q4" in ctx:
                        period = "quarter"
                    elif "H1" in ctx or "上半年" in ctx or "累计" in ctx or "YTD" in ctx:
                        period = "cumulative"
                    elif "全年" in ctx or "FY" in ctx:
                        period = "annual"
                    else:
                        period = "unknown"
                    seen.append((key, ctx, v, period))
        # Group by period.  Sign conflicts WITHIN the same period
        # are real disagreements (flag strongly); sign conflicts across
        # DIFFERENT periods are usually window-mismatches (flag with
        # window context so downstream can disambiguate).
        by_period: dict[str, list[tuple[str, str, float]]] = {}
        for key, ctx, v, p in seen:
            by_period.setdefault(p, []).append((key, ctx, v))

        # Cross-period conflict
        period_sums: dict[str, tuple[int, list[tuple[str, str, float]]]] = {}
        for p, items in by_period.items():
            if not items:
                continue
            dominant_sign = 1 if sum(1 for _, _, v in items if v >= 0) >= sum(1 for _, _, v in items if v < 0) else -1
            period_sums[p] = (dominant_sign, items)
        unique_signs = {s for s, _ in period_sums.values()}
        if len(unique_signs) > 1:
            for p, (_, items) in period_sums.items():
                for key, ctx, v in items:
                    findings.append(
                        f"- {metric_name} ({p}): {key}={v:+.2f}%  ctx=\"{ctx[:60]}\""
                    )

        # Same-period conflict (rare — flag strongly)
        for p, items in by_period.items():
            if len(items) < 2:
                continue
            signs = {1 if v >= 0 else -1 for _, _, v in items}
            if len(signs) > 1:
                for key, ctx, v in items:
                    findings.append(
                        f"- {metric_name} ({p}, SAME-PERIOD SIGN CONFLICT): "
                        f"{key}={v:+.2f}%  ctx=\"{ctx[:60]}\""
                    )
    if not findings:
        return ""
    return (
        "Same metric appears with OPPOSITE signs across reports. If "
        "different time windows (June monthly vs H1 cumulative), both "
        "may be correct — verify before citing. Downstream debators "
        "MUST resolve the window before quoting any of these figures.\n"
        + "\n".join(findings[:8])
    )
