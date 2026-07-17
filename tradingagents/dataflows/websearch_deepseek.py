"""DeepSeek-powered web search dataflow.

Thin Python port of the ``websearch-deepseek`` npm package's core logic: calls
DeepSeek's Anthropic-compatible API with the ``web_search_20250305`` tool type,
which triggers server-side web search and returns an AI-generated summary plus
source URLs.

No Node.js / MCP dependency — just an HTTP POST framed as per the documented
DeepSeek API contract. Needs ``DEEPSEEK_API_KEY`` (or ``WEBSEARCH_API_KEY``) in
the environment to work; degrades gracefully when absent.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_MODEL = "deepseek-v4-flash"
MAX_TOKENS = 32768
REQUEST_TIMEOUT = 60

# Which env vars to check — same precedence as the npm package.
_KEY_CANDIDATES = ("DEEPSEEK_API_KEY", "WEBSEARCH_API_KEY")


def _resolve_api_key() -> str | None:
    for var in _KEY_CANDIDATES:
        val = os.environ.get(var)
        if val:
            return val
    return None


# ── Public API ────────────────────────────────────────────────────────────


def web_search(query: str, **kwargs: Any) -> str:
    """Search the web via DeepSeek's native web search capability.

    Parameters
    ----------
    query : str
        The search query (natural language, be specific).
    **kwargs :
        Pass ``base_url``, ``model``, ``thinking`` (``"enabled"`` / ``"disabled"``)
        to override the defaults.

    Returns
    -------
    str
        Markdown-formatted search result: an AI-generated summary followed by
        source URLs.  If the key is missing or the API call fails, the return
        value explains what went wrong so the consuming LLM never needs to
        guess.
    """
    api_key = _resolve_api_key()
    if not api_key:
        return (
            "**Web search unavailable.**\n\n"
            "No DeepSeek API key detected.  "
            "Set ``DEEPSEEK_API_KEY`` (or ``WEBSEARCH_API_KEY``) in your "
            "environment to enable live web search for analysts."
        )

    base_url = kwargs.get("base_url", DEFAULT_BASE_URL)
    model = kwargs.get("model", os.environ.get("WEBSEARCH_MODEL", DEFAULT_MODEL))
    thinking = kwargs.get("thinking", "disabled")  # default off for speed

    url = f"{base_url}/v1/messages"
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a web search assistant. "
                    "Use web_search to find relevant, up-to-date information. "
                    "Then write a comprehensive, well-structured answer in plain text. "
                    "Include specific details, dates, and facts. "
                    "Answer in the same language as the query. "
                    "Do NOT output tool-call XML. "
                    "Do NOT call web_search again after you have results."
                ),
            },
            {"role": "user", "content": query},
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": {"type": "auto"},
    }
    if thinking == "enabled":
        body["thinking"] = {"type": "enabled"}

    try:
        resp = requests.post(
            url,
            headers={"content-type": "application/json", "x-api-key": api_key},
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.warning("DeepSeek web search network error: %s", exc)
        return f"**Web search network error.**\n\n{exc}"

    if not resp.ok:
        logger.warning("DeepSeek web search API error: HTTP %s", resp.status_code)
        return (
            f"**Web search API error (HTTP {resp.status_code}).**\n\n"
            f"{resp.text[:1000]}"
        )

    data = resp.json()
    content_blocks = data.get("content", [])
    if not content_blocks:
        return f'Web search for **"{query}"** returned no results.'

    results: list[dict] = []
    text_parts: list[str] = []

    for block in content_blocks:
        if block.get("type") == "web_search_tool_result":
            inner = block.get("content", [])
            for item in inner:
                if item.get("type") == "web_search_result":
                    results.append(
                        {
                            "title": item.get("title", "Untitled"),
                            "url": item.get("url", ""),
                            "page_age": item.get("page_age"),
                        }
                    )
        elif block.get("type") == "text":
            text = (block.get("text") or "").strip()
            if text:
                text_parts.append(text)

    if not results and not text_parts:
        return f'Web search for **"{query}"** returned no results.'

    lines: list[str] = []

    # AI-generated summary
    if text_parts:
        answer = "\n\n".join(text_parts)
        if answer.startswith("#") or answer.startswith("##"):
            lines.append(answer)
        else:
            lines.append(f"## Search Results: {query}")
            lines.append("")
            lines.append(answer)

    # Source URLs
    if results:
        if text_parts:
            lines.append("")
            lines.append("---")
        lines.append("")
        lines.append(f"### Sources ({len(results)}):")
        lines.append("")
        for i, r in enumerate(results, 1):
            line = f"{i}. [{r['title']}]({r['url']})"
            if r.get("page_age"):
                line += f"  — *{r['page_age']}*"
            lines.append(line)

    return "\n".join(lines)
