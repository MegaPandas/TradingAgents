"""End-to-end smoke for the simultaneous-debate primitive against a real LLM.

Drives the research debate (Bull vs Bear, 2 simultaneous rounds → Research
Manager) and the risk debate (Aggressive vs Conservative, 2 simultaneous
rounds → Neutral synthesizer) DIRECTLY — no LangGraph — so we can verify the
new order-neutral, delta-only design produces sane output on a given provider.

Simultaneity is simulated exactly as the graph does it: in each round BOTH
advocates read the SAME prior state (neither sees the other's current-round
output), then both turns are appended to the transcript (the operator.add
reducer does this in the real graph).

Usage:
    OPENAI_API_KEY=... python scripts/smoke_debate.py openai
    DEEPSEEK_API_KEY=... python scripts/smoke_debate.py deepseek
    ANTHROPIC_API_KEY=... python scripts/smoke_debate.py anthropic

This script does NOT call propagate(); it exercises only the debate nodes.
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.utils.debate_helpers import render_transcript, turns_for_side
from tradingagents.llm_clients import create_llm_client

PROVIDER_DEFAULTS = {
    "openai": "gpt-5.4-mini",
    "google": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen3.7-plus",
    "glm": "glm-5",
    "xai": "grok-4.3",
}


def _base_state(ticker: str) -> dict:
    """Minimal but realistic state for the debate nodes (no graph needed)."""
    return {
        "company_of_interest": ticker,
        "asset_type": "stock",
        "trade_date": "2026-07-18",
        "instrument_context": f"Instrument: {ticker} — sample company, tech sector.",
        "messages": [],
        "report_digest": (
            "MARKET: price 100, above 10/50SMA, below 200SMA, RSI 59, MACD bull cross.\n"
            "FUNDAMENTALS: TTM EPS 3.0, BVPS 25, ROE 15%, revenue growth 20%, "
            "Growth Quality 22/40. Fair-value range low 55 / median 78 / high 92.\n"
            "NEWS: June sales +5.5% YoY (first break of decline); overseas +94.7%.\n"
            "SENTIMENT: guba 5.5:1 bull/bear, score 6.8.\n"
            "MACRO: LPR 3.0%/3.5% flat, M2 8.0%, PMI 50.3."
        ),
        "fair_value_block": (
            "Fair Value (5-model, TTM anchor): Gordon 79 / Graham 273 (outlier) / "
            "RI 58 / PEG 71 / Justified PB 58. Range low 55 / median 78 / high 92. "
            "Normalized-EPS median 85."
        ),
        "latest_close": 100.0,
        "latest_close_settled_at": "2026-07-17",
        "latest_close_realtime": 100.0,
        "latest_close_realtime_at": "2026-07-18 15:00",
        "research_debate_turns": [],
        "risk_debate_turns": [],
    }


def _print(title: str, body: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}\n{body}")


def run_simultaneous_round(state: dict, node_a, node_b, turns_key: str) -> None:
    """One simultaneous round: both advocates read the SAME prior state, then
    both turns append to the transcript (mirrors the graph's parallel fan-out
    + operator.add reducer)."""
    out_a = node_a({**state})
    out_b = node_b({**state})
    state[turns_key] = list(state.get(turns_key, [])) + out_a.get(turns_key, []) + out_b.get(turns_key, [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=list(PROVIDER_DEFAULTS.keys()))
    parser.add_argument("--model", default=None, help="Override model (default per provider)")
    parser.add_argument("--rounds", type=int, default=2, help="Simultaneous rounds per debate")
    args = parser.parse_args()

    model = args.model or PROVIDER_DEFAULTS[args.provider]
    print(f"Provider: {args.provider} | Model: {model} | Rounds: {args.rounds}")

    client = create_llm_client(provider=args.provider, model=model)
    llm = client.get_llm()

    bull = create_bull_researcher(llm)
    bear = create_bear_researcher(llm)
    rm = create_research_manager(llm)
    aggressive = create_aggressive_debator(llm)
    conservative = create_conservative_debator(llm)
    neutral = create_neutral_debator(llm)

    ticker = "SAMPLE"
    state = _base_state(ticker)

    # --- Research debate: `rounds` simultaneous rounds, then Research Manager ---
    for r in range(args.rounds):
        run_simultaneous_round(state, bull, bear, "research_debate_turns")
        _print(f"Research debate — after round {r + 1}",
               render_transcript(state["research_debate_turns"], ("bull", "bear")))

    rm_out = rm({**state})
    state["investment_plan"] = rm_out["investment_plan"]
    _print("Research Manager — investment_plan", state["investment_plan"])

    # --- Risk debate: `rounds` simultaneous rounds, then Neutral synthesizer ---
    for r in range(args.rounds):
        run_simultaneous_round(state, aggressive, conservative, "risk_debate_turns")
        _print(f"Risk debate — after round {r + 1}",
               render_transcript(state["risk_debate_turns"], ("aggressive", "conservative")))

    neutral_out = neutral({**state})
    state["risk_synthesis"] = neutral_out["risk_synthesis"]
    _print("Neutral — risk_synthesis", state["risk_synthesis"])

    # --- Structure checks (the design invariants, best-effort) ---
    print("\n" + "=" * 70 + "\nStructure checks\n" + "=" * 70)
    failures = 0
    rturns = state["research_debate_turns"]
    kturns = state["risk_debate_turns"]

    def check(label, ok):
        nonlocal failures
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        failures += int(not ok)

    # Order-neutrality: each round has BOTH sides (neither ran first/last alone).
    for r in range(1, args.rounds + 1):
        sides_in_round = {t["side"] for t in rturns if t.get("round") == r}
        check(f"research round {r} has both sides (simultaneous)", sides_in_round == {"bull", "bear"})
    for r in range(1, args.rounds + 1):
        sides_in_round = {t["side"] for t in kturns if t.get("round") == r}
        check(f"risk round {r} has both sides (simultaneous)", sides_in_round == {"aggressive", "conservative"})

    # Round budget honored.
    check(f"bull wrote {args.rounds} turns", len(turns_for_side(rturns, "bull")) == args.rounds)
    check(f"bear wrote {args.rounds} turns", len(turns_for_side(rturns, "bear")) == args.rounds)
    check(f"aggressive wrote {args.rounds} turns", len(turns_for_side(kturns, "aggressive")) == args.rounds)
    check(f"conservative wrote {args.rounds} turns", len(turns_for_side(kturns, "conservative")) == args.rounds)

    # Synthesizers produced non-empty output.
    check("Research Manager produced investment_plan", bool(state.get("investment_plan", "").strip()))
    check("Neutral produced risk_synthesis", bool(state.get("risk_synthesis", "").strip()))

    # Delta-only heuristic: round-2 deltas are non-empty (advocates engaged).
    bull_r2 = [t for t in rturns if t.get("side") == "bull" and t.get("round") == 2]
    check("bull round-2 delta non-empty", bool(args.rounds < 2 or (bull_r2 and bull_r2[0].get("delta", "").strip())))

    # Neutral cites probabilities (EV discipline) — loose check.
    check("Neutral mentions a probability figure", "%" in state.get("risk_synthesis", ""))

    print()
    if failures:
        print(f"Smoke FAILED: {failures} structure check(s) failed.")
        return 1
    print("Smoke PASSED: simultaneous-debate structure invariants hold for", args.provider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
