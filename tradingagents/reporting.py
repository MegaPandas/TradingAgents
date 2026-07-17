"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

import re
from datetime import datetime
from pathlib import Path


# Risk-debate analysts tag every round of their argument with a literal
# ``Speaker:`` prefix when appending to history; use that to split the
# accumulated history into per-round sections so a multi-round (e.g. Deep
# depth = 5 rounds) report doesn't render as one concatenated blob.
_ROUND_SPLIT_RE = re.compile(r"(?:^|\n)(?=(?:Aggressive|Conservative|Neutral) Analyst:)")


def _render_debater_history(history: str, speaker_label: str) -> str:
    """Render a risk-debate history as one numbered block per debate round.

    The history field accumulates ``"Speaker: <argument>"`` segments across
    rounds. When research depth > 1 the same speaker runs multiple times, so
    we split on the ``Speaker:`` prefix and number each occurrence in order.

    ``speaker_label`` documents which speaker's history is being rendered;
    unused at the moment but kept on the signature so callers pass it
    explicitly and the call site stays self-documenting.
    """
    del speaker_label  # currently the prefix pattern covers all three labels
    if not history:
        return ""
    # Split on speaker prefix; each segment starts with "Speaker: ...".
    parts = [p.strip() for p in _ROUND_SPLIT_RE.split(history) if p.strip()]
    if len(parts) <= 1:
        return history.strip()
    rendered = []
    for idx, part in enumerate(parts, start=1):
        rendered.append(f"#### Round {idx}\n\n{part}")
    return "\n\n".join(rendered)


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if final_state.get("macro_policy_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "macro_policy.md").write_text(final_state["macro_policy_report"], encoding="utf-8")
        analyst_parts.append(("Macro & Policy Analyst", final_state["macro_policy_report"]))
    if final_state.get("situation_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "situation.md").write_text(final_state["situation_report"], encoding="utf-8")
        analyst_parts.append(("Situation Analyst", final_state["situation_report"]))
    if final_state.get("business_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "business.md").write_text(final_state["business_report"], encoding="utf-8")
        analyst_parts.append(("Business Analyst", final_state["business_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")


    # 4. Risk Management (display III)
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            aggressive_rendered = _render_debater_history(risk["aggressive_history"], "Aggressive Analyst")
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(aggressive_rendered, encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", aggressive_rendered))
        if risk.get("conservative_history"):
            conservative_rendered = _render_debater_history(risk["conservative_history"], "Conservative Analyst")
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(conservative_rendered, encoding="utf-8")
            risk_parts.append(("Conservative Analyst", conservative_rendered))
        if risk.get("neutral_history"):
            neutral_rendered = _render_debater_history(risk["neutral_history"], "Neutral Analyst")
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(neutral_rendered, encoding="utf-8")
            risk_parts.append(("Neutral Analyst", neutral_rendered))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## III. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager (display IV — final decision)
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## IV. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")


    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
