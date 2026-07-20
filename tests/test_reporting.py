"""Report parity: the shared writer produces the report tree for the CLI and the
programmatic API alike (#1037)."""

from types import SimpleNamespace

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree


def _state():
    return {
        "market_report": "MKT",
        "news_report": "NEWS",
        "research_debate_turns": [
            {"round": 1, "side": "bull", "delta": "BULL1"},
            {"round": 1, "side": "bear", "delta": "BEAR1"},
        ],
        "investment_plan": "RM PLAN",
        "risk_debate_turns": [
            {"round": 1, "side": "aggressive", "delta": "AGG1"},
            {"round": 1, "side": "conservative", "delta": "CON1"},
        ],
        "risk_synthesis": "NEUTRAL SYNTH",
        "final_trade_decision": "PM DECISION",
    }


@pytest.mark.unit
def test_write_report_tree_creates_files(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    assert out.name == "complete_report.md"
    assert (tmp_path / "1_analysts" / "market.md").read_text() == "MKT"
    assert (tmp_path / "1_analysts" / "news.md").read_text() == "NEWS"
    assert (tmp_path / "2_research" / "manager.md").read_text() == "RM PLAN"
    assert "BULL1" in (tmp_path / "2_research" / "bull.md").read_text()
    assert "BEAR1" in (tmp_path / "2_research" / "bear.md").read_text()
    assert (tmp_path / "4_risk" / "neutral.md").read_text() == "NEUTRAL SYNTH"
    assert "AGG1" in (tmp_path / "4_risk" / "aggressive.md").read_text()
    assert (tmp_path / "5_portfolio" / "decision.md").read_text() == "PM DECISION"
    complete = out.read_text()
    assert "Trading Analysis Report: AAPL" in complete
    assert "MKT" in complete and "PM DECISION" in complete


@pytest.mark.unit
def test_save_reports_explicit_path(tmp_path):
    # Unbound: with an explicit save_path, the method doesn't touch self/config.
    out = TradingAgentsGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert (tmp_path / "complete_report.md").exists()
    assert out == tmp_path / "complete_report.md"


@pytest.mark.unit
def test_save_reports_defaults_under_results_dir(tmp_path):
    mock_self = SimpleNamespace(config={"results_dir": str(tmp_path)})
    out = TradingAgentsGraph.save_reports(mock_self, _state(), "AAPL")
    assert out.exists()
    assert out.parent.parent.name == "reports"  # results_dir/reports/AAPL_<stamp>/...
    assert out.parent.name.startswith("AAPL_")
