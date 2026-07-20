"""End-to-end smoke for structured-output agents against a real LLM provider.

Runs the two structured-output decision agents (Research Manager, Portfolio
Manager) directly with their structured-output bindings and prints the
rendered markdown for each. Use this to verify a provider's native
structured-output mode (json_schema for OpenAI / xAI / DeepSeek / Qwen / GLM,
response_schema for Gemini, tool-use for Anthropic) returns clean instances
on the schemas we ship.

Usage:
    OPENAI_API_KEY=... python scripts/smoke_structured_output.py openai
    GOOGLE_API_KEY=... python scripts/smoke_structured_output.py google
    ANTHROPIC_API_KEY=... python scripts/smoke_structured_output.py anthropic
    DEEPSEEK_API_KEY=... python scripts/smoke_structured_output.py deepseek

The script does NOT call propagate(); it exercises only the two structured-
output decision calls plus the heuristic SignalProcessor.
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.llm_clients import create_llm_client

PROVIDER_DEFAULTS = {
    "openai": ("gpt-5.4-mini", None),
    "google": ("gemini-3.5-flash", None),
    "anthropic": ("claude-sonnet-4-6", None),
    "deepseek": ("deepseek-v4-flash", None),
    "qwen": ("qwen3.7-plus", None),
    "glm": ("glm-5", None),
    "xai": ("grok-4.3", None),
}


# Minimal but realistic state for the two decision agents (new transcript shape).
RESEARCH_DEBATE_TURNS = [
    {"round": 1, "side": "bull", "delta": (
        "Bull Analyst: NVDA data-center revenue grew 60% YoY last quarter on the "
        "Blackwell ramp; sovereign AI deals add a $40B+ multi-year tailwind.")},
    {"round": 1, "side": "bear", "delta": (
        "Bear Analyst: Concentration risk — top three customers >40% of revenue; "
        "any hyperscaler capex pause compresses the multiple.")},
]

RISK_SYNTHESIS = (
    "Neutral synthesis: bear 30% / base 30% / bull 40%; EV = 88.9 (-4.8% vs price 100). "
    "Balanced sizing: trim to 0.5-0.8x base, keep core for H2 demand-pull."
)


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "messages": [],
        "research_debate_turns": RESEARCH_DEBATE_TURNS,
        "report_digest": (
            "MARKET: NVDA above 10/50SMA, RSI 59.\n"
            "FUNDAMENTALS: TTM EPS 3.0, BVPS 25, ROE 15%, FV range 55-92.\n"
            "NEWS: data-center demand strong; export restrictions a cap."
        ),
    }


def _make_pm_state(investment_plan: str):
    return {
        "company_of_interest": "NVDA",
        "messages": [],
        "past_context": "",
        "risk_synthesis": RISK_SYNTHESIS,
        "investment_plan": investment_plan,
        "fair_value_block": (
            "Fair value (5-model, TTM anchor): low 55 / median 78 / high 92. "
            "Normalized-EPS median 85."
        ),
        "latest_close": 100.0,
        "latest_close_settled_at": "2026-07-17",
        "latest_close_realtime": 100.0,
        "latest_close_realtime_at": "2026-07-18 15:00",
    }


def _print_section(title: str, content: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}\n{content}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=list(PROVIDER_DEFAULTS.keys()))
    parser.add_argument("--deep-model", default=None, help="Override deep_think_llm")
    parser.add_argument("--quick-model", default=None, help="Override quick_think_llm")
    args = parser.parse_args()

    default_model, _ = PROVIDER_DEFAULTS[args.provider]
    deep_model = args.deep_model or default_model
    quick_model = args.quick_model or default_model

    print(f"Provider: {args.provider}")
    print(f"Deep model:  {deep_model}")
    print(f"Quick model: {quick_model}")

    deep_client = create_llm_client(provider=args.provider, model=deep_model)
    deep_llm = deep_client.get_llm()

    # 1) Research Manager (reads the research-debate transcript)
    rm = create_research_manager(deep_llm)
    rm_result = rm(_make_rm_state())
    investment_plan = rm_result["investment_plan"]
    _print_section("[1] Research Manager — investment_plan", investment_plan)

    # 2) Portfolio Manager (consumes RM plan + Neutral's risk_synthesis + FV)
    pm = create_portfolio_manager(deep_llm)
    pm_result = pm(_make_pm_state(investment_plan))
    final_decision = pm_result["final_trade_decision"]
    _print_section("[2] Portfolio Manager — final_trade_decision", final_decision)

    # 3) SignalProcessor extracts the rating with zero LLM calls.
    sp = SignalProcessor()
    rating = sp.process_signal(final_decision)
    _print_section("[3] SignalProcessor → rating", rating)

    # 4) Lightweight checks: each rendered output should carry the expected
    #    section headers so downstream consumers (memory log, CLI display,
    #    saved reports) keep working.
    checks = [
        ("Research Manager", investment_plan, ["**Recommendation**:"]),
        ("Portfolio Manager", final_decision, ["**Rating**:", "**Executive Summary**:", "**Investment Thesis**:"]),
    ]
    print("\n" + "=" * 70 + "\nStructure checks\n" + "=" * 70)
    failures = 0
    for name, text, required in checks:
        for marker in required:
            ok = marker in text
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: contains {marker!r}")
            failures += int(not ok)

    print()
    if failures:
        print(f"Smoke FAILED: {failures} structure check(s) missing.")
        return 1
    print("Smoke PASSED: structured output → rendered markdown chain works for", args.provider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
