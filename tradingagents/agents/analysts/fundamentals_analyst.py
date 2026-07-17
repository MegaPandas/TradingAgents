from tradingagents.agents.utils.agent_utils import (
    _PIPELINE_PREAMBLE,
    chat_prompt_messages,
    format_fair_value_block,
    format_price_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            _PIPELINE_PREAMBLE + """ROLE
You are the Valuation Analyst. Estimate the fair value of the instrument and derive a justified target price from the financial statements and peer comparables. All figures are in the instrument's quote currency (CNY for China A-shares).

SCOPE (cover every item)
0. **The deterministic 5-model fair value range is in your DATA block below** (Gordon Growth / Graham / Residual Income / PEG / Justified PB). Cite the range verbatim. DO NOT call any tool to recompute it — use the pre-computed values as your valuation anchor. Layer manual cross-checks on top.
1. Earnings power — revenue trend; gross/operating/net margin trajectory; normalized diluted EPS (latest period and TTM). Flag one-off items inside net income.
2. Balance-sheet quality — leverage (debt-to-equity, interest cover), ROE, ROIC, working-capital changes.
3. Cash flow — operating cash flow conversion (OCF/NI), capex, free cash flow yield.
4. Relative valuation — PE / PB / PS / EV-EBITDA against (a) sector peers and (b) the stock's own 5-year range. State the peer set and justify it.
4b. GROWTH-ADJUSTED multiples (mandatory for high-growth names):
    - PEG ratio = forward PE / forward EPS growth rate (%). PEG < 1 = undervalued, 1-2 = fair, > 2 = expensive.
    - EV / forward-revenue-growth — how many dollars of EV per dollar of expected growth.
    - Price / sales-to-growth (PSG) — for early-stage or pre-profit names.
    - If revenue 3yr CAGR ≥ 25%, ALWAYS show the PEG-based fair value alongside PE-based.
5. Absolute valuation — run a DCF only if FCF is positive and reasonably stable; otherwise use an earnings-power or Gordon-growth cross-check. State every assumption (growth rate, discount rate, terminal multiple).
5b. GROWTH-STAGE DCF (when forward FCF is negative or unreliable):
    - Stage 1 (Y1-5): apply consensus revenue-growth rate; revenue × assumed FCF margin to get FCF.
    - Stage 2 (Y6-10): fade growth linearly to terminal growth.
    - Stage 3 (terminal): sector-median EV/EBITDA × terminal EBITDA.
    - Two scenarios (bull / base) with explicit weights. State every assumption.
6. Per-share discipline — use diluted share count; separate current-year from next-year EPS; never mix periods.

7. GROWTH QUALITY SCORE (mandatory output, 0-40 total):
    a. Revenue durability (0-10): TAM growth, market-share trajectory, recurring-revenue %, customer-concentration risk.
    b. Margin trajectory (0-10): gross/operating margin trend over 3 years (expanding/flat/compressing).
    c. Capital efficiency (0-10): ROIC trend, capex / revenue ratio, working-capital intensity.
    d. Moat (0-10): pricing power evidence, switching costs, tech / regulatory / scale barrier.
    Final verdict: "quality growth" (≥ 30) / "speculative growth" (20-29) / "no growth premium justified" (< 20).

HARD CONSTRAINTS
- Every target price must name its method and the assumptions driving it. No target without a shown basis.
- If a required input is missing, output "cannot value — missing X" and state what is missing. Do not fabricate a peer multiple or a growth rate.
- Cite the source tool output for every number. When two tools disagree, flag the conflict; do not invent a reconciled value.
- Re-state the latest close when quoting upside/downside.
- For high-growth names (revenue CAGR ≥ 25%), the PEG-based or growth-stage DCF result MUST be shown alongside PE-based. A bare PE verdict on a high-growth name is not acceptable.
- **PRICE INTEGRITY**: the latest_close value in the prompt is the TRUE split-adjusted (qfq) current price. Do NOT multiply, divide, or adjust it for stock splits, share consolidations, reverse splits, or any capital event — the pre-fetched value is already corrected. If you find stock-split data in a financial statement and are tempted to re-derive the price, IGNORE IT and use latest_close as given.

OUTPUT
- Methodology and key assumptions
- Peer set (tickers or sector-median multiples used)
- Growth-adjusted fair value (PEG / EV-sales-growth / growth-stage DCF when applicable)
- Fair-value range: low / base / high
- Target price (state the EPS period it rests on) + upside/downside vs latest close
- Verdict: undervalued / fairly valued / overvalued
- Growth Quality Score (4 dimensions, total /40, plus verdict)
- Append a Markdown table summarizing the key valuation inputs."""
            + get_language_instruction()
        )

        # Move price + fair value context to data_block so the static system
        # template stays byte-identical across runs → full prompt-cache hit.
        data_block_parts = [format_price_context(state), format_fair_value_block(state)]
        data_block = "\n".join(p for p in data_block_parts if p)

        prompt = chat_prompt_messages(
            system_message, tools,
            current_date, instrument_context,
            data_block=data_block,
        )

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
