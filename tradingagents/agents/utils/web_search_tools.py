"""LangChain tools for live web search, wrapping the DeepSeek web search dataflow.

``web_search(query)`` — analyst calls freely at runtime to fact-check, find
current industry news, or explore any topic not covered by pre-fetched data.

The tool degrades gracefully when ``DEEPSEEK_API_KEY`` is absent: the return
value explains why search is unavailable, so the LLM never invents an answer.
"""
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.websearch_deepseek import web_search as _search


@tool
def web_search(
    query: Annotated[
        str,
        "Search query — natural language, be specific. "
        "Include context like date, sector, or company name for better results. "
        "Examples: 'BYD 固态电池 2026 试产进展', 'China NEV subsidy policy 2026 July', "
        "'特斯拉 Q2 交付量 2026 超预期'.",
    ],
) -> str:
    """Search the web for current information on any topic.

    Uses DeepSeek's native web search (server-side execution). Returns an
    AI-generated summary of findings plus source URLs. Best for:

    - Fact-checking claims from training data against current sources
    - Finding real-time industry news, policy changes, or company events
    - Exploring topics not covered by pre-fetched data blocks

    Requires ``DEEPSEEK_API_KEY`` or ``WEBSEARCH_API_KEY`` to be set in the
    environment.  When the key is absent the tool returns a clear explanation
    rather than failing silently.

    Args:
        query: Free-text search query.

    Returns:
        Markdown string with AI-generated summary and source URLs.
    """
    return _search(query)
