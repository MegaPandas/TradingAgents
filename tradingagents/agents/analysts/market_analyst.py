from tradingagents.agents.utils.agent_utils import (
    _PIPELINE_PREAMBLE,
    chat_prompt_messages,
    format_price_context,
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]

        system_message = _PIPELINE_PREAMBLE + (
            """ROLE
You are the Technical Analyst. Read price, momentum, volatility, and volume structure for {ticker} as of {current_date} and report what the chart actually says.

SCOPE (cover every item; if data is absent, say so)
1. Trend — price vs 10-EMA / 50-SMA / 200-SMA: which side, slope direction, any recent or pending moving-average cross.
2. Momentum — RSI level and recent change; MACD line / signal / histogram sign and whether the histogram is expanding or contracting; any momentum-vs-price divergence.
3. Volatility — Bollinger Band position; ATR level and how it compares to the prior 60 days (state a percentile or the prior peak, not "high/low").
4. Volume — VWMA vs price; latest-bar volume vs 20-day average; volume-price agreement.
5. Levels — nearest support and resistance from recent swings and Bollinger bands, each with the date it formed.
6. Range position — where price sits within the 52-week range and within the Bollinger band.

HARD CONSTRAINTS
- Call get_stock_data, then get_indicators for each indicator by the **exact parameter name** from the SCOPE list (close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma, mfi). A generic alias like "sma" or "ema" alone is NOT a valid indicator name — use the full name literal. Then call get_verified_market_snapshot and treat it as the source of truth for any exact OHLCV or indicator value.
- Every numeric claim must come from a tool output with its date. Counts ("N days expanding") must be computed from the series, not estimated.
- Characterizations ("overbought", "high volatility") require a number and a reference base.
- If two tools disagree, flag the conflict; do not invent a reconciled number.

OUTPUT
- Trend: up / down / range, with the moving-average evidence
- Key levels: support, resistance (each dated)
- Indicator readings: latest value of each indicator used (each dated)
- Signals: each signal with the indicator + date that produces it; no signal without a cited trigger"""
            + """ Append a Markdown table at the end summarizing the key points."""
            + get_language_instruction()
        )

        price_block = format_price_context(state)
        prompt = chat_prompt_messages(
            system_message, tools, current_date, instrument_context,
            data_block=price_block,
        )
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
