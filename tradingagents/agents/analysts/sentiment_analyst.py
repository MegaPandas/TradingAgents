"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches four complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing
  4. 股吧 (eastmoney guba) — China retail-investor discussion board
                           (domestic counterpart to StockTwits/Reddit;
                           degrades to a placeholder for non-China symbols)

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    _PIPELINE_PREAMBLE,
    chat_prompt_messages,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.guba import fetch_guba_messages
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.symbol_utils import is_cn_share


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch all sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        # guba (eastmoney) is the China retail-sentiment source; it returns a
        # placeholder for non-CN symbols, so US/global runs are unaffected.
        # P1 dedup (D3): the news block is pre-fetched once at propagation
        # time and shared with the News analyst; fall back to get_news.func
        # only if the propagator did not populate it (e.g. older graphs).
        news_block = state.get("news_block")
        if not news_block:
            news_block = get_news.func(ticker, start_date, end_date)
        if is_cn_share(ticker):
            # China A-shares: guba is the retail-sentiment source. StockTwits
            # has no A-share coverage (HTTP 404) and r/stocks won't discuss a
            # 6-digit CN code — calling them only burns the Reddit 429 backoff
            # budget, so skip them outright for CN symbols.
            stocktwits_block = "<stocktwits: not applicable to China A-shares>"
            reddit_block = "<reddit: not applicable to China A-shares>"
            guba_block = fetch_guba_messages(ticker, limit=30)
        else:
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)
            guba_block = "<guba: not a China A-share>"

        # Fix 15: data in data_block, system message static → prompt-cache hit.
        # No bind_tools — data is pre-fetched in-node.
        data_block = _build_data_block(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            guba_block=guba_block,
        )

        prompt = chat_prompt_messages(
            _SENTIMENT_SYSTEM_TEMPLATE, tools=[],
            current_date=end_date, instrument_context=instrument_context,
            data_block=data_block,
        )

        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_data_block(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    guba_block: str,
) -> str:
    """Assemble dynamic sentiment data into the data_block format (Fix 15).

    Previously this was embedded in the system_message, breaking prompt cache.
    Now returned as a data_block string so the static _SENTIMENT_SYSTEM_TEMPLATE
    stays byte-identical across runs.
    """
    return f"""{get_language_instruction()}
===== SENTIMENT DATA (pre-fetched, instrument: {ticker}, window: {start_date} to {end_date}) =====
- News headlines (past 7 days)
- StockTwits / Reddit (not applicable to China A-shares — placeholders)
- 股吧 guba — China retail discussion (China shares only)

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

<start_of_reddit>
{reddit_block}
<end_of_reddit>

### 股吧 guba — China retail discussion (China shares only)
<start_of_guba>
{guba_block}
<end_of_guba>"""


_SENTIMENT_SYSTEM_TEMPLATE = _PIPELINE_PREAMBLE + """ROLE
You are the Sentiment Analyst. Read retail and social sentiment and report only what the fetched data supports.

SCOPE
1. Per-source reading — for each fetched source (股吧 guba for China shares; StockTwits + Reddit otherwise): post volume, bull/bear ratio or tone, standout posts with engagement.
2. Cross-source divergence — where sources disagree in direction.
3. Dominant narrative themes — the recurring topic across sources.
4. Catalysts and risks surfaced in the chatter.

HARD CONSTRAINTS
- Report only what the fetched blocks contain. If a source returned an "<unavailable>" or "<not applicable>" placeholder, state that explicitly and exclude it from the ratios; do not invent posts, counts, or ratios.
- Lower confidence when a source is missing or sparse (fewer than 5 data points).
- Distinguish opinion from event.

OUTPUT
- overall_band: one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish
- overall_score: 0–10 (5 = neutral)
- confidence: low / medium / high — driven by data availability, not conviction
- narrative: per-source breakdown with cited counts/engagement, divergences, themes, catalysts and risks, and a markdown summary table of key signals"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
