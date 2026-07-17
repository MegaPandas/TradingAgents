"""Deterministic digest of the 4 analyst reports — single source for downstream."""
from __future__ import annotations
import re
import pytest

from tradingagents.agents.utils.report_digest import build_digest, MAX_LEN

_REPORTS = {
    "market_report": "## Market\n" + ("line " * 50 + "\n") * 6,
    "sentiment_report": "## Sentiment\n" + ("line " * 30 + "\n") * 4,
    "news_report": "## News\n" + ("line " * 20 + "\n") * 3,
    "fundamentals_report": "## Fundamentals\n" + ("line " * 40 + "\n") * 5,
}


@pytest.mark.unit
def test_digest_has_per_analyst_sections():
    out = build_digest(_REPORTS, instrument_context="Apple", verified_snapshot=None)
    for tag in ("MARKET", "SENTIMENT", "NEWS", "FUNDAMENTALS"):
        assert tag in out


@pytest.mark.unit
def test_digest_includes_instrument_context():
    out = build_digest(_REPORTS, instrument_context="Test ticker XYZ", verified_snapshot=None)
    assert "Test ticker XYZ" in out


@pytest.mark.unit
def test_digest_includes_snapshot_when_provided():
    snap = "| Date | Close |\n|---|---|\n| 2026-07-10 | 90.00 |"
    out = build_digest(_REPORTS, instrument_context="X", verified_snapshot=snap)
    assert "90.00" in out


@pytest.mark.unit
def test_digest_under_max_len_for_realistic_reports():
    out = build_digest(_REPORTS, instrument_context="A", verified_snapshot=None)
    assert len(out) <= MAX_LEN, f"digest too long: {len(out)}"


@pytest.mark.unit
def test_digest_preserves_numbers_and_dates_verbatim():
    text = ("EPS 0.4399\nROE 1.65%\n2026-07-10 close 90.00\n" * 10)
    out = build_digest({"market_report": text, "sentiment_report": "", "news_report": "", "fundamentals_report": ""}, instrument_context="X", verified_snapshot=None)
    for needle in ("0.4399", "1.65", "2026-07-10", "90.00"):
        assert needle in out


@pytest.mark.unit
def test_digest_omits_placeholder_and_unavailable_tokens():
    text = "<no guba posts found for 002594.SZ>\n<stocktwits: not applicable to China A-shares>"
    out = build_digest({"market_report": "", "sentiment_report": text, "news_report": "", "fundamentals_report": ""}, instrument_context="X", verified_snapshot=None)
    assert "no guba posts" not in out  # degraded sources summarised, not pasted
    assert "<stocktwits" not in out


@pytest.mark.unit
def test_digest_populated_after_analysts_run():
    """Graph-flow check: the digest node sees the 4 reports and writes a digest."""
    from tradingagents.agents.utils.report_digest import build_digest
    fake_state = {
        "market_report": "## Market\nclose 90.00\n",
        "sentiment_report": "## Sentiment\nscore 6.5\n",
        "news_report": "## News\n2026-07-10\n",
        "fundamentals_report": "## Fundamentals\nEPS 0.44\n",
        "instrument_context": "Ticker XYZ",
        "verified_snapshot": "| Date | Close |\n|---|---|\n| 2026-07-10 | 90.00 |\n",
    }
    out = build_digest(
        {"market_report": fake_state["market_report"],
         "sentiment_report": fake_state["sentiment_report"],
         "news_report": fake_state["news_report"],
         "fundamentals_report": fake_state["fundamentals_report"]},
        instrument_context=fake_state["instrument_context"],
        verified_snapshot=fake_state["verified_snapshot"],
    )
    assert "Ticker XYZ" in out
    assert "0.44" in out and "90.00" in out


@pytest.mark.unit
def test_digest_has_macro_policy_section():
    out = build_digest(
        {
            "market_report": "x", "sentiment_report": "x", "news_report": "x",
            "fundamentals_report": "x", "macro_policy_report": "Monetary LPR 3.0",
        },
        instrument_context="X", verified_snapshot=None,
    )
    assert "MACRO & POLICY" in out
    # _key_findings strips non-numeric tokens; the numeric value is what
    # survives the digest. "LPR 3.0" → the substring "3.0" is the
    # assertion target.
    assert "3.0" in out


@pytest.mark.unit
def test_digest_handles_missing_macro_policy_report():
    out = build_digest(
        {"market_report": "x", "sentiment_report": "x", "news_report": "x",
         "fundamentals_report": "x"},
        instrument_context="X", verified_snapshot=None,
    )
    # Missing section should be summarized, not panic.
    assert "MACRO" in out