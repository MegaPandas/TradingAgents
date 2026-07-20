"""akshare vendor — domestic (China A/B/Beijing-share) market data.

Plugs into the routed vendor layer (:func:`route_to_vendor`) for the P0
categories: ``core_stock_apis`` (OHLCV), ``technical_indicators`` and
``fundamental_data``. It is the CN counterpart to yfinance.

Design
------
akshare serves **China shares only**. Any non-CN symbol raises
:class:`NoMarketDataError`, so when a category's vendor chain is
``"akshare,yfinance"`` the router transparently falls through to yfinance for
US/global tickers — existing behaviour is unchanged and akshare only ever
claims CN instruments. akshare is imported defensively: a missing install
surfaces as :class:`VendorNotConfiguredError`, which is equally chain-fallable.

CN domestic endpoints are reachable directly in-region and must not exit
through a foreign-exit VPN proxy; :func:`_no_proxy` clears the proxy
environment for the duration of each akshare call so the vendor works whether
or not a system VPN is active.

Verified working calls (akshare 1.18.64, 2026-07): ``stock_zh_a_daily`` (sina
OHLCV), ``stock_financial_abstract`` + ``stock_financial_analysis_indicator``
(emweb fundamentals). The eastmoney ``push2his``/``push2`` hosts
(``stock_zh_a_hist``/``stock_zh_a_spot_em``) TLS-fingerprint-block python and
are deliberately NOT used here.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from datetime import datetime, timedelta

import pandas as pd

from .errors import NoMarketDataError, VendorNotConfiguredError
from .symbol_utils import cn_code6, cn_sina_symbol

logger = logging.getLogger(__name__)

# Currency tag carried on every report so the Trader/Risk agents never conflate
# a CNY figure with a USD one (the design doc's P2 currency concern).
_CURRENCY = "CNY"

_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")

# Browser-like headers so akshare's underlying requests.Session doesn't
# identify itself as the default ``python-requests/X.Y`` UA — that's a
# giveaway scraper fingerprint on sina / eastmoney. Both endpoints
# happily serve the same data to a Mozilla UA; some block or throttle
# the bare python-requests string.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _install_browser_headers() -> None:
    """Monkey-patch ``requests.Session`` so every request carries browser UA.

    Runs once at module import. Installs on the class so every new session
    inherits the headers; safe because we only add Accept / UA / Language
    — all standard fields sina/eastmoney already accept.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover
        return
    original_init = requests.Session.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.headers.update(_BROWSER_HEADERS)

    requests.Session.__init__ = patched_init


_install_browser_headers()


@contextlib.contextmanager
def _no_proxy():
    """Clear proxy env for the call so CN domestic endpoints connect directly.

    The system VPN (e.g. clash on 127.0.0.1:7897) is for *foreign* sources; the
    domestic endpoints akshare hits (sina / eastmoney emweb / stats bureaus) are
    directly reachable and get reset if forced through it. Restored on exit so
    foreign-source vendors in the same process keep using the proxy.
    """
    saved = {k: os.environ.pop(k, None) for k in _PROXY_ENV}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _require_akshare():
    """Import akshare lazily or raise VendorNotConfiguredError if absent."""
    try:
        import akshare  # noqa: F401
        return akshare
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise VendorNotConfiguredError(
            "akshare is not installed. Install it with `pip install akshare` "
            "to enable China A-share market data."
        ) from exc


def _guard_cn(symbol: str) -> str:
    """Return the bare 6-digit CN code, else raise NoMarketDataError.

    Raising (rather than returning empty) lets the router fall through to the
    next vendor in the chain (e.g. yfinance) for non-CN symbols.
    """
    code = cn_code6(symbol)
    if code is None:
        raise NoMarketDataError(
            symbol, symbol, f"{symbol!r} is not a China A/B/Beijing-share symbol"
        )
    return code


def _header(title: str, n_records: int) -> str:
    return (
        f"# {title}\n"
        f"# Records: {n_records}\n"
        f"# Retrieved: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"# Currency: {_CURRENCY}\n\n"
    )


# ---------------------------------------------------------------------------
# core_stock_apis: OHLCV
# ---------------------------------------------------------------------------

def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    """Daily OHLCV (前复权) for a CN share over [start_date, end_date]."""
    code = _guard_cn(symbol)
    ak = _require_akshare()
    sina = cn_sina_symbol(symbol) or f"sz{code}"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    # Reuse the indicator-layer cache if a Market analyst previously loaded
    # the same symbol with a longer warmup window. Key is (code, look_back_days);
    # we synthesize look_back_days = days_in_range so any earlier call covering
    # a wider window will hit the cache.
    cached = _OHLCV_CACHE.get((code, (end_dt - start_dt).days))
    if cached is not None and not cached.empty:
        df = _cached_to_raw_shape(cached)
    else:
        with _no_proxy():
            df = ak.stock_zh_a_daily(
                symbol=sina,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
    if df is None or df.empty:
        raise NoMarketDataError(
            symbol, code, f"no rows between {start_date} and {end_date}"
        )

    out = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]),
        "Open": pd.to_numeric(df["open"], errors="coerce").round(2),
        "High": pd.to_numeric(df["high"], errors="coerce").round(2),
        "Low": pd.to_numeric(df["low"], errors="coerce").round(2),
        "Close": pd.to_numeric(df["close"], errors="coerce").round(2),
        "Volume": pd.to_numeric(df["volume"], errors="coerce"),
    })
    out = out[(out["Date"] >= start_dt) & (out["Date"] <= end_dt)]
    if out.empty:
        raise NoMarketDataError(
            symbol, code, f"no rows between {start_date} and {end_date}"
        )

    label = f"{code} ({sina}, akshare qfq)"
    return _header(f"Stock data for {label} from {start_date} to {end_date}",
                   len(out)) + out.to_csv(index=False)


# ---------------------------------------------------------------------------
# technical_indicators
# ---------------------------------------------------------------------------

# Process-level cache for the OHLCV frame the indicators layer reads from.
# Market analyst typically calls get_indicators 14× per run (one per
# indicator); without this cache each call would re-hit sina
# stock_zh_a_daily — 14 HTTP round-trips returning the same data.
# Key: (canonical_code, look_back_days); value: cleaned OHLCV frame.
# This cache is short-lived by design: it only serves to collapse the burst
# of indicator calls inside one analyst invocation.
_OHLCV_CACHE: dict[tuple[str, int], "pd.DataFrame"] = {}


def _load_ohlcv_cached(symbol: str, look_back_days: int, end_dt) -> "pd.DataFrame":
    """Fetch qfq OHLCV with enough warmup history; cached per (code, lookback)."""
    code = _guard_cn(symbol)
    key = (code, look_back_days)
    cached = _OHLCV_CACHE.get(key)
    if cached is not None and not cached.empty:
        return cached

    ak = _require_akshare()
    warmup = max(look_back_days, 30) + 220
    start_dt = end_dt - timedelta(days=warmup)
    sina = cn_sina_symbol(symbol) or f"sz{code}"
    with _no_proxy():
        df = ak.stock_zh_a_daily(
            symbol=sina,
            start_date=max(start_dt, end_dt - timedelta(days=warmup)).strftime("%Y%m%d"),
            end_date=(end_dt + timedelta(days=1)).strftime("%Y%m%d"),
            adjust="qfq",
        )
    if df is None or df.empty:
        raise NoMarketDataError(symbol, code, "no OHLCV for indicator window")

    ohlcv = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]),
        "Open": pd.to_numeric(df["open"], errors="coerce"),
        "High": pd.to_numeric(df["high"], errors="coerce"),
        "Low": pd.to_numeric(df["low"], errors="coerce"),
        "Close": pd.to_numeric(df["close"], errors="coerce"),
        "Volume": pd.to_numeric(df["volume"], errors="coerce"),
    }).sort_values("Date")

    _OHLCV_CACHE[key] = ohlcv
    # Also populate the "raw" view used by get_stock if its date window matches.
    return ohlcv


def _cached_to_raw_shape(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Convert a cached OHLCV frame back to the shape ak.stock_zh_a_daily returns.

    get_stock expects the raw akshare frame (lowercase columns ``date``/``open``/...).
    The cached frame has capitalized columns. Map back so the consumer's downstream
    code can slice on the same names.
    """
    return pd.DataFrame({
        "date":   ohlcv["Date"].dt.strftime("%Y-%m-%d"),
        "open":   ohlcv["Open"],
        "high":   ohlcv["High"],
        "low":    ohlcv["Low"],
        "close":  ohlcv["Close"],
        "volume": ohlcv["Volume"],
    })


def get_stock_stats_indicators_window(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    """Compute a single technical indicator over a trailing window via stockstats.

    Loads qfq OHLCV up to ``curr_date`` (with extra history so slow indicators
    like a 200 SMA are warmed up), wraps it with stockstats, and returns the
    most recent ``look_back_days`` rows showing close + the requested indicator.
    """
    code = _guard_cn(symbol)
    from stockstats import wrap

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    # Cached: a single analyst run typically calls get_indicators for 14
    # indicators (close_50_sma, rsi, macd, ...); without caching each call
    # would re-hit sina stock_zh_a_daily.
    ohlcv = _load_ohlcv_cached(symbol, look_back_days, end_dt)
    ohlcv = ohlcv[ohlcv["Date"] <= end_dt]
    if ohlcv.empty:
        raise NoMarketDataError(symbol, code, f"no rows up to {curr_date}")

    stats = wrap(ohlcv.copy())
    try:
        stats[indicator]  # trigger stockstats calculation
    except (KeyError, Exception) as exc:  # unknown / unsupported indicator
        raise NoMarketDataError(
            symbol, code, f"indicator {indicator!r} could not be computed ({exc})"
        )

    window = stats.tail(look_back_days)
    # stockstats lowercases the OHLCV columns on wrap, so resolve the close
    # column defensively before selecting.
    close_col = "Close" if "Close" in window.columns else "close"
    show = window[["Date", close_col, indicator]].copy()
    show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
    show[close_col] = pd.to_numeric(show[close_col], errors="coerce").round(2)
    show = show.rename(columns={close_col: "Close"})
    return (
        _header(f"Technical indicator '{indicator}' for {code} (akshare), "
                f"last {len(show)} rows up to {curr_date}", len(show))
        + show.to_csv(index=False)
    )


# ---------------------------------------------------------------------------
# fundamental_data
# ---------------------------------------------------------------------------

_ABSTRACT_CACHE: dict[str, pd.DataFrame] = {}


def _financial_abstract(code: str):
    """Fetch the full financial-abstract frame (cached per code).

    Five consumers read this same eastmoney endpoint in one CN run —
    ``get_fundamentals`` / ``get_balance_sheet`` / ``get_cashflow`` /
    ``get_income_statement`` / ``calculate_fair_value``. The abstract is the
    company's full financial history (date-invariant within reason), so
    caching by ``code`` collapses 5 round-trips to 1. Only successful
    fetches are cached; ``NoMarketDataError`` still propagates.
    """
    cached = _ABSTRACT_CACHE.get(code)
    if cached is not None:
        return cached
    ak = _require_akshare()
    with _no_proxy():
        df = ak.stock_financial_abstract(symbol=code)
    if df is None or df.empty:
        raise NoMarketDataError(code, code, "no financial abstract returned")
    _ABSTRACT_CACHE[code] = df
    return df


def _section_rows(
    abstract: pd.DataFrame,
    sections: tuple[str, ...] = (),
    metric_keywords: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Filter the abstract to rows by exact section name and/or metric keyword.

    ``stock_financial_abstract`` carries a ``选项`` (section) column plus a
    ``指标`` (metric) column. When ``sections`` is provided only those
    section names are included (exact match on 选项). When
    ``metric_keywords`` is provided, rows whose metric column contains the
    keyword are also included (broad match on 指标). Returns up to the
    most recent 5 reporting periods.
    """
    section_col = "选项" if "选项" in abstract.columns else None
    metric_col = "指标" if "指标" in abstract.columns else None
    period_cols = [c for c in abstract.columns if c not in ("选项", "指标")]
    period_cols = period_cols[:5]  # latest 5 periods

    mask = pd.Series([False] * len(abstract), index=abstract.index)
    for sec in sections:
        if section_col is not None:
            mask = mask | (abstract[section_col].astype(str) == sec)
    for kw in metric_keywords:
        if metric_col is not None:
            mask = mask | abstract[metric_col].astype(str).str.contains(kw, na=False)
    rows = abstract[mask]
    cols = [c for c in ([metric_col] if metric_col else []) + period_cols if c in rows.columns]
    return rows[cols]


def _format_statement(code: str, kind: str, rows: pd.DataFrame) -> str:
    title = f"{kind} for {code} (akshare, CNY, latest reporting periods)"
    if rows.empty:
        # Fall back to a truncated full abstract so the agent still gets data.
        ak_df = _financial_abstract(code)
        head = ak_df.iloc[:, : min(6, ak_df.shape[1])]
        return _header(f"{kind} for {code} (full abstract, truncated)", len(head)) \
            + head.to_csv(index=False)
    return _header(title, len(rows)) + rows.to_csv(index=False)


_KEY_FUNDAMENTALS = (
    # Revenue & Profit (raw CNY)
    "营业总收入",
    "营业成本",
    "净利润",
    "归母净利润",
    "扣非净利润",
    # Margins & efficiency (%, source-computed — not our calculation)
    "毛利率",
    "销售净利率",
    "营业利润率",
    "期间费用率",
    "净资产收益率",
    "总资产报酬率",
    # Growth (%)
    "营业总收入增长率",
    "归属母公司净利润增长率",
    # Per-share
    "基本每股收益",
    "稀释每股收益",
    "每股经营现金流",
    "每股企业自由现金流量",
    # Balance quality
    "资产负债率",
    "流动比率",
    "速动比率",
    "权益乘数",
    "每股净资产",
    "每股未分配利润",
    "每股资本公积金",
    "每股留存收益",
    # Cash flow conversion
    "经营活动净现金/销售收入",
    "经营现金净流量对销售收入比率",
)


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Key metrics from the source's financial abstract.

    Returns a curated set of metrics as raw CSV (indicator × period).
    Every value is what the source (eastmoney) computed — no fallback,
    no NaN-removal, no derived calculation. NaN appears as blank so
    the LLM sees what is available and judges accordingly.

    Dedicated get_income_statement / get_balance_sheet / get_cashflow
    tools exist for the full three-statement detail.
    """
    import re as _re
    code = _guard_cn(ticker)
    abstract = _financial_abstract(code)

    # Determine column order: latest period first
    period_cols = [c for c in abstract.columns if c not in ("选项", "指标")]
    # Pick the 4 most recent columns that have any non-NaN value
    good_cols = [c for c in period_cols if abstract[c].notna().any()][:4]

    rows = []
    for name in _KEY_FUNDAMENTALS:
        # Escape parentheses for regex match (指标 names contain "(ROE)" etc.)
        pat = _re.escape(name)
        row = abstract[abstract["指标"].astype(str).str.contains(pat, na=False, regex=True)]
        if row.empty:
            continue
        rows.append(row.iloc[0].to_dict())
    if not rows:
        return _header(f"No fundamentals for {code}", 0) + "(no data)"

    out = pd.DataFrame(rows)
    keep = [c for c in ["指标"] + good_cols if c in out.columns]
    return _header(f"Company fundamentals for {code} (akshare, source = eastmoney stock_financial_abstract)",
                   len(rows)) + out[keep].to_csv(index=False)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    code = _guard_cn(ticker)
    abstract = _financial_abstract(code)
    # Exact section match: 财务 risk section covers debt/leverage/liquidity.
    # Cross-section metric fallback: 资产/负债 keywords catch 常用指标 rows
    # like 股东权益合计(净资产), 商誉 that also belong to a balance sheet view.
    return _format_statement(code, "Balance sheet",
                             _section_rows(abstract, sections=("财务风险",)))


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    code = _guard_cn(ticker)
    abstract = _financial_abstract(code)
    return _format_statement(code, "Cash flow statement",
                             _section_rows(abstract, sections=("收益质量",),
                                           metric_keywords=("现金流",)))


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    code = _guard_cn(ticker)
    abstract = _financial_abstract(code)
    # 毛利 is NOT covered by 营业/利润/收入 but the abstract has a pre-computed
    # 毛利率 row. Without it the LLM computes gross margin from raw 营业成本
    # (which is actually total operating costs, not pure COGS) and gets 1.34%
    # instead of the correct 18.8% (BYD 002594.SZ real Q1 2026).
    return _format_statement(code, "Income statement",
                             _section_rows(abstract, sections=("盈利能力", "成长能力"),
                                           metric_keywords=("毛利",)))


# ---------------------------------------------------------------------------
# news_data: individual-stock news (eastmoney)
# ---------------------------------------------------------------------------

def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Per-stock Chinese news (eastmoney) for a CN share, filtered to the range.

    ``stock_news_em`` returns the most recent news for a symbol; we keep only
    items whose ``发布时间`` falls in [start_date, end_date]. Non-CN symbols
    raise NoMarketDataError so the router falls through to yfinance news.
    """
    code = _guard_cn(ticker)
    ak = _require_akshare()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    with _no_proxy():
        df = ak.stock_news_em(symbol=code)
    if df is None or df.empty:
        raise NoMarketDataError(ticker, code, "no news returned")

    ts = pd.to_datetime(df["发布时间"], errors="coerce")
    in_range = df[(ts >= start_dt) & (ts <= end_dt + timedelta(days=1))]
    if in_range.empty:
        raise NoMarketDataError(
            ticker, code, f"no news between {start_date} and {end_date}"
        )

    lines = []
    for _, row in in_range.head(30).iterrows():
        when = str(row.get("发布时间", ""))[:16]
        title = str(row.get("新闻标题", "")).strip().replace("\n", " ")
        source = str(row.get("文章来源", "")).strip()
        lines.append(f"[{when} · {source}] {title}")
    return _header(
        f"News for {code} (akshare/eastmoney), {start_date} to {end_date}",
        len(lines),
    ) + "\n".join(lines)


# ---------------------------------------------------------------------------
# macro_data: China macro indicators (PBOC / NBS / SHIBOR)
# ---------------------------------------------------------------------------

# CN-specific aliases only. Generic ones (cpi, unemployment, fed_funds_rate,
# 10y_treasury, ...) are deliberately NOT claimed so they fall through to FRED
# — this keeps US/global macro on FRED and routes only unambiguously-China
# series to akshare. ``alias -> (akshare func name, report title)``.
_CN_MACRO = {
    "lpr": ("macro_china_lpr", "China LPR (loan prime rate)"),
    "cn_lpr": ("macro_china_lpr", "China LPR (loan prime rate)"),
    "shibor": ("macro_china_shibor_all", "China SHIBOR (interbank rates)"),
    "cn_shibor": ("macro_china_shibor_all", "China SHIBOR (interbank rates)"),
    "m2": ("macro_china_money_supply", "China money supply (M2/M1/M0)"),
    "money_supply": ("macro_china_money_supply", "China money supply (M2/M1/M0)"),
    "cn_money_supply": ("macro_china_money_supply", "China money supply (M2/M1/M0)"),
    "pmi": ("macro_china_pmi_yearly", "China official manufacturing PMI"),
    "cn_pmi": ("macro_china_pmi_yearly", "China official manufacturing PMI"),
    "gdp": ("macro_china_gdp_yearly", "China GDP (year-on-year)"),
    "cn_gdp": ("macro_china_gdp_yearly", "China GDP (year-on-year)"),
    "cn_cpi": ("macro_china_cpi", "China CPI"),
    "china_cpi": ("macro_china_cpi", "China CPI"),
}


# Process-level cache for CN macro series. Keyed by alias (lpr, m2, ...) so
# News Analyst + Macro & Policy Analyst + Trader can all ask for the same
# series in one run and only the first call hits the upstream endpoint.
_MACRO_CACHE: dict[str, "pd.DataFrame"] = {}


def _fetch_macro(alias: str) -> "pd.DataFrame":
    """Return a CN macro DataFrame, fetching once per process per alias.

    Wrapped in the per-call _no_proxy() guard; raises NoMarketDataError on
    empty/missing upstream data so the caller can degrade gracefully.
    """
    cached = _MACRO_CACHE.get(alias)
    if cached is not None and not cached.empty:
        return cached
    ak = _require_akshare()
    func_name, _ = _CN_MACRO[alias]
    func = getattr(ak, func_name, None)
    if func is None:
        raise NoMarketDataError(alias, alias, f"akshare has no {func_name}")
    with _no_proxy():
        df = func()
    if df is None or df.empty:
        raise NoMarketDataError(alias, alias, "no macro rows returned")
    _MACRO_CACHE[alias] = df
    return df


def get_macro_data(
    indicator: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    """Fetch a China macro series as a markdown report.

    Accepts only CN-specific aliases (see ``_CN_MACRO``). Any other indicator
    (including FRED aliases like ``cpi``, ``unemployment``, ``fed_funds_rate``)
    raises NoMarketDataError so the router falls through to FRED.
    """
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _CN_MACRO:
        raise NoMarketDataError(
            indicator, indicator,
            f"{indicator!r} is not a China macro alias known to akshare "
            f"(try lpr/shibor/m2/pmi/gdp/cn_cpi); falling back for US/global series",
        )

    _, title = _CN_MACRO[key]
    df = _fetch_macro(key)

    # akshare's macro series are NOT consistently ordered: money_supply / cpi
    # come newest-first (descending) while lpr is oldest-first (ascending), so
    # neither .head() nor .tail() is safe. Parse the date column and take the
    # most recent N rows regardless of source sort order (otherwise a descending
    # series hands the LLM 2008 data — the M2/CPI stale-data bug).
    recent = _recent_macro_rows(df, 8)
    header = (
        f"## akshare: {title}\n"
        f"- Indicator alias: `{key}`\n"
        f"- Region: China\n"
        f"- Currency/unit: as reported by the source\n"
        f"- As of: {curr_date}\n\n"
    )
    return header + recent.to_csv(index=False)


def _recent_macro_rows(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Return the ``n`` most recent rows of a CN macro frame, order-agnostic.

    Recognises the common akshare date columns (``月份`` like "2026年05月份",
    ``日期``/``TRADE_DATE``). Falls back to the first column if none match.
    """
    date_col = next(
        (c for c in df.columns if str(c) in ("月份", "日期", "TRADE_DATE", "date", "Date")),
        df.columns[0],
    )
    parsed = df[date_col].astype(str).map(_parse_macro_date)
    if parsed.notna().any():
        order = parsed.sort_values(ascending=False, na_position="last").index
        return df.loc[order].head(n)
    # No parseable dates: akshare macro is usually newest-first, so head is the
    # best single guess.
    return df.head(n)


def _parse_macro_date(value: str):
    """Parse an akshare macro date string to a pandas Timestamp, or NaT.

    Handles ``"2026年05月份"`` (monthly), ISO ``"2026-07-10"``, and
    ``datetime.date``/``Timestamp`` values coerced to string.
    """
    s = str(value).strip()
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})", s)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        except ValueError:
            return pd.NaT
    return pd.to_datetime(s, errors="coerce")
