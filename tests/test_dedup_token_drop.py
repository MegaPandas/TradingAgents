"""P1 acceptance: input-token count drops and content coverage holds."""
from tradingagents.agents.utils.report_digest import build_digest


def _fake_state():
    long_block = ("line of analyst text\n" * 50) * 4   # ~1 KB per report
    return {
        "market_report": "## Market\n" + long_block,
        "sentiment_report": "## Sentiment\n" + long_block,
        "news_report": "## News\n" + long_block,
        "fundamentals_report": "## Fundamentals\n" + long_block,
    }


def test_digest_token_drop():
    """Digest size < 25% of the 4-report concatenation."""
    s = _fake_state()
    big = s["market_report"] + s["sentiment_report"] + s["news_report"] + s["fundamentals_report"]
    digest = build_digest(
        {"market_report": s["market_report"], "sentiment_report": s["sentiment_report"],
         "news_report": s["news_report"], "fundamentals_report": s["fundamentals_report"]},
        instrument_context="Ticker XYZ", verified_snapshot=None,
    )
    assert len(digest) < 0.25 * len(big), f"digest too large: {len(digest)} vs {len(big)}"


def test_digest_preserves_key_numbers():
    text = "EPS 0.4399\nROE 1.65%\n2026-07-10 close 90.00\n"
    out = build_digest({"market_report": text, "sentiment_report": "", "news_report": "", "fundamentals_report": ""},
                       instrument_context="X", verified_snapshot=None)
    for needle in ("0.4399", "1.65", "2026-07-10", "90.00"):
        assert needle in out
