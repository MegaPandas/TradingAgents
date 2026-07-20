"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

from datetime import datetime
from pathlib import Path


def _side_text(turns: list[dict], side: str) -> str:
    """Concatenate one advocate's turns (round-labelled) from the debate transcript."""
    parts = []
    for t in turns:
        if t.get("side") == side:
            parts.append(f"**Round {t.get('round')}**\n{t.get('delta', '')}")
    return "\n\n".join(parts)


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

    # 2. Research (simultaneous debate transcript + Research Manager synthesis)
    research_turns = final_state.get("research_debate_turns") or []
    investment_plan = final_state.get("investment_plan") or ""
    if research_turns or investment_plan:
        research_dir = save_path / "2_research"
        research_parts = []
        bull_text = _side_text(research_turns, "bull")
        if bull_text:
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(bull_text, encoding="utf-8")
            research_parts.append(("Bull Researcher", bull_text))
        bear_text = _side_text(research_turns, "bear")
        if bear_text:
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(bear_text, encoding="utf-8")
            research_parts.append(("Bear Researcher", bear_text))
        if investment_plan:
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(investment_plan, encoding="utf-8")
            research_parts.append(("Research Manager", investment_plan))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 4. Risk Management (display III — simultaneous debate transcript + Neutral synthesis)
    risk_turns = final_state.get("risk_debate_turns") or []
    risk_synthesis = final_state.get("risk_synthesis") or ""
    if risk_turns or risk_synthesis:
        risk_dir = save_path / "4_risk"
        risk_parts = []
        aggressive_text = _side_text(risk_turns, "aggressive")
        if aggressive_text:
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(aggressive_text, encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", aggressive_text))
        conservative_text = _side_text(risk_turns, "conservative")
        if conservative_text:
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(conservative_text, encoding="utf-8")
            risk_parts.append(("Conservative Analyst", conservative_text))
        if risk_synthesis:
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk_synthesis, encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk_synthesis))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## III. Risk Management Team Decision\n\n{content}")

    # 5. Portfolio Manager (display IV — final decision)
    final_trade_decision = final_state.get("final_trade_decision") or ""
    if final_trade_decision:
        portfolio_dir = save_path / "5_portfolio"
        portfolio_dir.mkdir(exist_ok=True)
        (portfolio_dir / "decision.md").write_text(final_trade_decision, encoding="utf-8")
        sections.append(f"## IV. Portfolio Manager Decision\n\n### Portfolio Manager\n{final_trade_decision}")


    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
