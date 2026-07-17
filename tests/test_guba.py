"""Unit tests for the eastmoney 股吧 (guba) sentiment fetcher."""

from unittest.mock import patch

import pytest

from tradingagents.dataflows import guba

# A minimal slice of a guba list page: two post rows with the four fields the
# parser extracts (title / read / reply / update).
_SAMPLE_HTML = """
<html><body>
<div class="articleh">
 <span class="title"><a data-postid="111" data-posttype="0" href="/news,002594,111.html">比亚迪大涨创新高</a></span>
 <span class="read">1796</span><span class="reply">15</span><span class="update">07-09 11:41</span>
</div>
<div class="articleh">
 <span class="title"><a data-postid="222" data-posttype="0" href="/news,002594,222.html">果然是大盘逆子</a></span>
 <span class="read">201</span><span class="reply">1</span><span class="update">07-09 02:13</span>
</div>
</body></html>
"""


class _FakeResp:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.unit
def test_parses_titles_and_metrics():
    with patch.object(guba, "urlopen", return_value=_FakeResp(_SAMPLE_HTML)):
        out = guba.fetch_guba_messages("002594.SZ", limit=5)
    assert "比亚迪大涨创新高" in out
    assert "果然是大盘逆子" in out
    assert "1796" in out          # read count
    assert "07-09 11:41" in out   # update timestamp
    assert "guba (eastmoney)" in out


@pytest.mark.unit
def test_non_cn_returns_placeholder_without_request():
    out = guba.fetch_guba_messages("AAPL")
    assert "not a China A-share" in out


@pytest.mark.unit
def test_network_error_degrades_to_placeholder():
    with patch.object(guba, "urlopen", side_effect=OSError("connection reset")):
        out = guba.fetch_guba_messages("002594.SZ")
    assert "unavailable" in out


@pytest.mark.unit
def test_empty_board_returns_placeholder():
    with patch.object(guba, "urlopen", return_value=_FakeResp("<html>no posts</html>")):
        out = guba.fetch_guba_messages("002594.SZ")
    assert "no guba posts" in out
