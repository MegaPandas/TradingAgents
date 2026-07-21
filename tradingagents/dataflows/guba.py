"""东方财富股吧 (eastmoney guba) sentiment fetcher — China retail-investor posts.

The domestic counterpart to StockTwits/Reddit for A-share retail sentiment.
eastmoney's per-stock 股吧 is public (no login) and very high-volume: every
A/B/Beijing share has a board at ``guba.eastmoney.com/list,{code}.html``.

Like the other sentiment fetchers it degrades to a placeholder string and never
raises, so the sentiment analyst (which calls StockTwits + Reddit + 股吧 in
parallel) is never special-cased for missing data.
"""
from __future__ import annotations

import http.client
import logging
import math
import re
from urllib.request import Request, urlopen

from .symbol_utils import cn_code6

logger = logging.getLogger(__name__)

_URL = "https://guba.eastmoney.com/list,{code}.html"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_TITLE_RE = re.compile(r'class="title"><a[^>]*data-postid="\d+"[^>]*>([^<]+)</a>')
_READ_RE = re.compile(r'class="read">([0-9,]+)')
_REPLY_RE = re.compile(r'class="reply">([0-9,]+)')
_UPDATE_RE = re.compile(r'class="update">([^<]+)')

# Threshold for flagging a single post as an outlier: if one post accounts
# for >40% of total reads or >40% of total replies, flag it so the LLM
# knows the aggregate signal is dominated by one post.
_OUTLIER_READ_PCT = 0.40
_OUTLIER_REPLY_PCT = 0.40


def _to_int(s: str) -> int | None:
    try:
        return int(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _interaction_weight(reads: int | None, replies: int | None) -> float:
    """Log-scale interaction score: log10(1 + reads) + log10(1 + replies).

    Compresses viral posts (100k reads ≈ score 5) and dead posts (0 reads ≈ 0)
    into a narrow range so no single post dominates a weighted aggregate.
    """
    r = math.log10(1 + (reads or 0))
    p = math.log10(1 + (replies or 0))
    return round(r + p, 2)


def fetch_guba_messages(ticker: str, limit: int = 60, timeout: float = 10.0) -> str:
    """Fetch recent 股吧 posts for a CN share and return a plaintext block.

    Non-CN symbols get a placeholder (the sentiment analyst calls all sources in
    parallel; each degrades independently). Never raises — the caller gets a
    uniform string interface regardless of success.

    The output now includes an interaction-weighted statistics header that helps
    the LLM detect when a single outlier post dominates the aggregate signal
    (common on low-volume boards where one deep-analysis post gets most reads).
    """
    code = cn_code6(ticker)
    if code is None:
        return f"<guba unavailable: {ticker!r} is not a China A-share>"

    req = Request(_URL.format(code=code), headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException) as exc:
        logger.warning("guba fetch failed for %s: %s", ticker, exc)
        return f"<guba unavailable: {type(exc).__name__}>"

    titles = _TITLE_RE.findall(html)
    if not titles:
        return f"<no guba posts found for {ticker}>"
    reads = _READ_RE.findall(html)
    replies = _REPLY_RE.findall(html)
    updates = _UPDATE_RE.findall(html)

    total_read = total_reply = 0
    total_weight = 0.0
    max_read = 0
    max_reply = 0
    max_read_title = ""
    max_reply_title = ""
    lines = []
    for i, title in enumerate(titles[:limit]):
        read = _to_int(reads[i]) if i < len(reads) else None
        reply = _to_int(replies[i]) if i < len(replies) else None
        upd = updates[i].strip() if i < len(updates) else "?"
        weight = _interaction_weight(read, reply)

        if read is not None:
            total_read += read
            if read > max_read:
                max_read = read
                max_read_title = title.strip()
        if reply is not None:
            total_reply += reply
            if reply > max_reply:
                max_reply = reply
                max_reply_title = title.strip()
        total_weight += weight

        r = str(read) if read is not None else "-"
        c = str(reply) if reply is not None else "-"
        lines.append(
            f"[{upd} · read {r} · reply {c} · iw {weight}] {title.strip()}"
        )

    # --- Statistics header ---
    header_parts = [
        f"guba (eastmoney) for {ticker}: {len(lines)} recent posts",
        f"total reads {total_read} · total replies {total_reply}",
        f"total interaction-weight {total_weight:.1f} (log-scale, higher = more engaged board)",
    ]

    # Outlier detection: flag when one post dominates attention
    outliers = []
    if total_read > 0 and max_read / total_read > _OUTLIER_READ_PCT:
        pct = round(100 * max_read / total_read)
        outliers.append(
            f"⚠ OUTLIER POST (reads): \"{max_read_title[:80]}\" accounts for "
            f"{pct}% of all reads ({max_read}/{total_read}). "
            f"Do NOT let this single post dominate the sentiment signal."
        )
    if total_reply > 0 and max_reply / total_reply > _OUTLIER_REPLY_PCT:
        pct = round(100 * max_reply / total_reply)
        outliers.append(
            f"⚠ OUTLIER POST (replies): \"{max_reply_title[:80]}\" accounts for "
            f"{pct}% of all replies ({max_reply}/{total_reply}). "
            f"Weigh carefully — high replies may indicate controversy, not consensus."
        )

    header = "\n".join(header_parts)
    if outliers:
        header += "\n\n" + "\n".join(outliers)

    return header + "\n\n" + "\n".join(lines)
