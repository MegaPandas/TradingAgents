"""Tests for the shared analyst-report concatenation helper.

``collect_analyst_reports`` is the single fallback used by the researchers
and debators when ``report_digest`` is absent (bare test states). Keeping it
in one place stops the 7-way string concat from being copy-pasted.
"""

import pytest

from tradingagents.agents.utils.agent_utils import collect_analyst_reports


@pytest.mark.unit
class TestCollectAnalystReports:
    def test_concatenates_present_reports_in_canonical_order(self):
        state = {
            "market_report": "M",
            "sentiment_report": "S",
            "news_report": "N",
            "fundamentals_report": "F",
            "macro_policy_report": "P",
            "business_report": "B",
            "situation_report": "U",
        }
        assert collect_analyst_reports(state) == "MSNFPBU"

    def test_missing_reports_default_to_empty(self):
        # Only market_report present — the rest contribute "".
        assert collect_analyst_reports({"market_report": "M"}) == "M"

    def test_empty_state_returns_empty_string(self):
        assert collect_analyst_reports({}) == ""

    def test_order_matches_canonical_pipeline(self):
        # market → sentiment → news → fundamentals → macro_policy → business → situation
        state = {
            "situation_report": "U",
            "market_report": "M",
            "fundamentals_report": "F",
            "sentiment_report": "S",
            "business_report": "B",
            "news_report": "N",
            "macro_policy_report": "P",
        }
        # Insertion order in the dict is intentionally scrambled; output must be canonical.
        assert collect_analyst_reports(state) == "MSNFPBU"
