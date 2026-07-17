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


def _to_int(s: str) -> int | None:
    try:
        return int(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def fetch_guba_messages(ticker: str, limit: int = 30, timeout: float = 10.0) -> str:
    """Fetch recent 股吧 posts for a CN share and return a plaintext block.

    Non-CN symbols get a placeholder (the sentiment analyst calls all sources in
    parallel; each degrades independently). Never raises — the caller gets a
    uniform string interface regardless of success.
    """
    code = cn_code6(ticker)
    if code is None:
        return f"<guba unavailable: {ticker!r} is not a China A-share>"

    req = Request(_URL.format(code=code), headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors.
        logger.warning("guba fetch failed for %s: %s", ticker, exc)
        return f"<guba unavailable: {type(exc).__name__}>"

    titles = _TITLE_RE.findall(html)
    if not titles:
        return f"<no guba posts found for {ticker}>"
    reads = _READ_RE.findall(html)
    replies = _REPLY_RE.findall(html)
    updates = _UPDATE_RE.findall(html)

    total_read = total_reply = 0
    lines = []
    for i, title in enumerate(titles[:limit]):
        read = _to_int(reads[i]) if i < len(reads) else None
        reply = _to_int(replies[i]) if i < len(replies) else None
        upd = updates[i].strip() if i < len(updates) else "?"
        if read is not None:
            total_read += read
        if reply is not None:
            total_reply += reply
        r = str(read) if read is not None else "-"
        c = str(reply) if reply is not None else "-"
        lines.append(f"[{upd} · read {r} · reply {c}] {title.strip()}")

    summary = (
        f"guba (eastmoney) for {ticker}: {len(lines)} recent posts · "
        f"total reads {total_read} · total replies {total_reply}"
    )
    return summary + "\n\n" + "\n".join(lines)
