from tradingagents.agents.utils.agent_utils import (
    _PIPELINE_PREAMBLE,
    chat_prompt_messages,
    safe_llm_invoke,
    get_global_news,
    get_insider_transactions,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    web_search,
)
from tradingagents.dataflows.symbol_utils import is_cn_share


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        ticker = state["company_of_interest"]
        cn = is_cn_share(ticker)
        instrument_context = get_instrument_context_from_state(state)

        # P1 dedup (D3): if the propagation step pre-fetched the news block,
        # inject it into the prompt and drop get_news from the tool list so
        # the LLM consumes the shared fetch instead of re-invoking the tool.
        news_block = state.get("news_block")
        tools = [get_global_news, get_macro_indicators, web_search, get_insider_transactions]
        if news_block:
            news_clause = (
                f"### Pre-fetched news (past 7 days)\n"
                f"<start_of_news>\n{news_block}\n<end_of_news>\n"
            )
        else:
            tools.append(get_news)
            news_clause = ""
        # get_prediction_markets removed entirely (Fix 3 + audit) —
        # Polymarket is unreachable from China and the tool was
        # removed from the News ToolNode.

        if cn:
            macro_clause = (
                "get_macro_indicators(indicator, curr_date, look_back_days) to "
                "ground macro commentary in actual **China** data via akshare — "
                "use the CN aliases 'lpr' (loan prime rate), 'shibor' (interbank "
                "rates), 'm2' (money supply), 'pmi' (manufacturing PMI), 'gdp', "
                "or 'cn_cpi'"
            )
        else:
            macro_clause = (
                "get_macro_indicators(indicator, curr_date, look_back_days) to "
                "ground macro commentary in actual data from FRED (e.g. 'cpi', "
                "'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', "
                "'yield_curve')"
            )

        # Only mention get_news in the tool list when it is actually bound (no pre-fetch).
        if news_block:
            tools_clause = (
                f"get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news"
            )
        else:
            tools_clause = (
                f"get_news(ticker, start_date, end_date) for {asset_label}-specific news, "
                f"get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news"
            )

        # Static system_message — byte-identical across all runs → prompt cache hit.
        # Dynamic clauses (tools_clause, macro_clause, news_clause) moved to
        # data_block so the system-message template stays cacheable (Fix 13).
        data_clauses = f"""TOOL INSTRUCTIONS (dynamic — varies per market/run):
- {tools_clause}
- {macro_clause}
- web_search(query): live search via DeepSeek for current facts, industry news, policy changes, competitive developments not covered by pre-fetched data.
{news_clause}"""

        system_message = _PIPELINE_PREAMBLE + (
            "ROLE\n"
            "You are the News Analyst. Identify the information flow that actually moves the instrument — stock-specific events, sector developments, and the dated catalysts ahead — and separate fact from speculation.\n\n"
            "SCOPE (cover every item)\n"
            "1. Stock-specific events — filings, earnings, product, M&A, management, litigation, supply-chain, that directly name the instrument.\n"
            "2. Sector / industry developments that affect the revenue or cost line. When analyzing a China A-share, call get_global_news with CN macro or sector keywords (e.g. 'China PMI M2 GDP auto industry') to capture up-to-date industry context beyond what the ticker-specific news provides. When in doubt about a factual claim or when you need current details on a specific topic (e.g. 'BYD 固态电池 2026 试产进展', 'China NEV subsidy 2026'), call web_search(query) to retrieve live information.\n"
            "3. Upcoming dated catalysts — next earnings date, product launch, policy effective date, lock-up expiry.\n"
            "4. Materiality — for each item, state whether it is priced in, partially priced, or not yet reflected, and why.\n"
            "5. Fact vs speculation — label each item as reported fact, analyst speculation, or market chatter.\n\n"
            "HARD CONSTRAINTS\n"
            "- Cite source outlet and timestamp for every item. No item without a fetched source.\n"
            "- Do not paraphrase a number into existence; quote the figure from the tool output.\n"
            "- Use web_search(query) to verify any key factual claim from training data before publishing. If web_search returns 'unavailable', state that explicitly and do not fabricate sources.\n"
            "- **TIME-WINDOW DISCIPLINE (Fix 7)**: for periodic figures (sales, deliveries, revenue), ALWAYS state the period in the same line as the number. '6月销量同比 +5.46%' ≠ 'H1累计销量同比 -15.72%' (different time windows). Downstream reports (Business, Situation, Bear, Conservative) WILL conflict if you omit the window. Use one of: 'X月单月 YoY', 'H1累计 YoY', 'YTD YoY', 'Q1-Qn YoY'. Numbers without a period label are rejected.\n"
            "OUTPUT\n"
            "- Key events: dated, sourced, labeled fact/speculation, with materiality\n"
            "- Upcoming catalysts: dated\n"
            "- Sector context relevant to the instrument\n"
            "- Net implication for the current position on the instrument: one-sentence summary of how the news flow changes the near-term outlook (bullish/bearish/neutral, with the single strongest piece of evidence)\n"
            "- Append a Markdown table summarizing the key events.\n\n"
            "Your TOOL INSTRUCTIONS (which tools to use and how) are in the DATA block below — read them before calling any tool."
            + get_language_instruction()
        )

        prompt = chat_prompt_messages(
            system_message, tools, current_date, instrument_context,
            data_block=data_clauses,
        )

        chain = prompt | llm.bind_tools(tools)
        result = safe_llm_invoke(chain, state["messages"], "News Analyst")

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
