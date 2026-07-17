"""Symbol normalization and market-data error types for vendor calls.

Yahoo Finance (the default vendor) uses specific ticker conventions that
differ from the broker / TradingView / MT5 style symbols users often type:

    user types        Yahoo wants       why
    ---------------   ---------------   -----------------------------------
    XAUUSD, XAUUSD+   GC=F              gold has no forex pair on Yahoo;
                                        it is quoted as a COMEX future
    EURUSD            EURUSD=X          spot forex pairs take a ``=X`` suffix
    BTCUSD            BTC-USD           crypto pairs use a ``-`` separator
    SPX500, US500     ^GSPC             index CFDs map to Yahoo index symbols

Passing the raw broker symbol to Yahoo returns an empty result, which the
agents previously received as free text and could hallucinate a price
around (see issue #781). Centralizing the mapping here means every yfinance
entry point resolves symbols the same way, and new instruments are added by
appending a table row rather than editing call sites.
"""

from __future__ import annotations

import logging
import re

# NoMarketDataError lives in the vendor-error taxonomy (errors.py); re-exported
# here for the many call sites that import it alongside normalize_symbol.
from .errors import NoMarketDataError as NoMarketDataError

logger = logging.getLogger(__name__)


# ISO-4217 codes common enough to appear in retail forex pairs. A bare
# six-letter symbol whose halves are BOTH in this set is treated as a spot
# forex pair and given Yahoo's ``=X`` suffix.
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

# Crypto bases that brokers quote against USD without a separator.
_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

# Explicit aliases for instruments whose broker symbol does not map to a
# Yahoo symbol by rule. Metals/energy resolve to their front-month future;
# index CFD names resolve to the underlying Yahoo index symbol. Extend by
# adding rows — no call site changes required.
_ALIASES = {
    # Precious metals (spot names -> COMEX/NYMEX futures)
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs -> Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


# Crypto quote currencies that all map to Yahoo's USD pair. Yahoo lists only
# ``<BASE>-USD`` (not the USDT/USDC stablecoin pairs), so a broker symbol quoted
# in any of these resolves to ``-USD`` (#982). Longest first so ``USDT``/``USDC``
# match before the ``USD`` substring.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")


def crypto_base(raw: str) -> str | None:
    """Return the crypto base (e.g. ``BTC``) for a known USD/USDT/USDC-quoted
    crypto symbol in any form the pipeline may hold — ``BTC-USD``, ``BTCUSD``,
    ``BTC-USDT`` — or None for non-crypto symbols. Purely syntactic.
    """
    if not isinstance(raw, str):
        return None
    compact = raw.strip().upper().rstrip("+").replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            return base if base in _CRYPTO_BASES else None
    return None


def _normalize_crypto(s: str) -> str | None:
    """Return ``<BASE>-USD`` for a known USD/USDT/USDC-quoted crypto, else None."""
    base = crypto_base(s)
    return f"{base}-USD" if base else None


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol.

    Resolution order (first match wins):
      1. Explicit alias table (metals, energy, index CFDs).
      2. Crypto rule: a known crypto base quoted in USD/USDT/USDC (dashed or
         not) -> ``BASE-USD``.
      3. Forex rule: six letters that are two ISO currency codes -> ``PAIR=X``.
      4. Otherwise the upper-cased symbol is returned unchanged (plain
         equities, ETFs, Yahoo-native symbols like ``GC=F`` or ``^GSPC``).

    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching. The function is purely syntactic — it performs no network
    calls — so it is safe to apply on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Broker CFD/qualifier suffixes Yahoo never uses.
    s = s.rstrip("+")

    crypto = _normalize_crypto(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters Yahoo symbols use."""
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None


# ---------------------------------------------------------------------------
# China A/B-share (and Beijing exchange) symbols.
#
# Users type Yahoo-style suffixed symbols (``002594.SZ``, ``600519.SS``) so the
# existing Yahoo path keeps working; each domestic vendor needs a different
# shape, so these helpers produce the per-vendor canonical form from any input.
#
#   user types        akshare-sina      akshare-emweb / guba    xueqiu
#   ---------------   ---------------   ----------------------  ----------
#   002594.SZ         sz002594          002594                  SZ002594
#   600519.SS         sh600519          600519                  SH600519
#
# Exchange is inferred from the suffix when present, else from the leading
# digits of a bare 6-digit code (main board + STAR + ChiNext + Beijing).
# ---------------------------------------------------------------------------

_CN_SUFFIX_EXCHANGE = {".SZ": "SZ", ".SH": "SH", ".BJ": "BJ"}
# Bare 6-digit routing by first digit(s). Good enough for the boards retail
# traders analyze; 9xxxxx/900xxx (Shanghai B) map to SH, 200xxx (Shenzhen B)
# to SZ, 8/4/920 (Beijing) to BJ.
_CN_LEADING_EXCHANGE = {
    "6": "SH", "9": "SH",           # Shanghai main / B
    "0": "SZ", "2": "SZ", "3": "SZ",  # Shenzhen main / B / ChiNext
    "8": "BJ", "4": "BJ",            # Beijing Stock Exchange
}


def cn_code6(raw: str) -> str | None:
    """Return the bare 6-digit A/B-share code (``"002594"``) or None.

    Accepts ``002594``, ``002594.SZ``, ``SZ002594``, ``sz002594``. Purely
    syntactic — no network calls.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip().upper().lstrip("+")
    # strip an exchange prefix if present
    for pre in ("SZ", "SH", "BJ"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    # strip a Yahoo-style CN suffix only (.SH/.SS/.SZ/.BJ); leave arbitrary
    # suffixes (e.g. BRK.B, 002594.XX) intact so they are not misread as CN.
    # .SS accepted as legacy alias for .SH (Yahoo historical convention).
    for suf in (".SH", ".SS", ".SZ", ".BJ"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s if s.isdigit() and len(s) == 6 else None


def is_cn_share(raw: str) -> bool:
    """True when ``raw`` looks like a China A/B/Beijing-share symbol.

    Relies on ``cn_code6``, which handles all standard forms (bare 6-digit,
    ``.SZ`` / ``.SS`` / ``.BJ`` suffix, ``SZ`` / ``SH`` prefix). A suffix-only
    check is deliberately avoided — it would return True for non-CN symbols
    accidentally ending in ``.SZ`` (e.g. ``BRK.B.SZ``).
    """
    return cn_code6(raw) is not None


def _cn_exchange(raw: str) -> str | None:
    """Resolve the exchange tag (``SH``/``SZ``/``BJ``) for a CN symbol."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    for suf, ex in _CN_SUFFIX_EXCHANGE.items():
        if s.endswith(suf):
            return ex
    for pre, ex in (("SZ", "SZ"), ("SH", "SH"), ("BJ", "BJ")):
        if s.startswith(pre):
            return ex
    code = cn_code6(raw)
    if code:
        return _CN_LEADING_EXCHANGE.get(code[0])
    return None


def cn_sina_symbol(raw: str) -> str | None:
    """akshare sina-backend form: lowercase prefixed ``sz002594`` / ``sh600519``."""
    code = cn_code6(raw)
    ex = _cn_exchange(raw)
    if not code or not ex:
        return None
    return f"{ex.lower()}{code}"


def cn_xueqiu_symbol(raw: str) -> str | None:
    """xueqiu form: uppercase prefixed ``SZ002594`` / ``SH600519``."""
    code = cn_code6(raw)
    ex = _cn_exchange(raw)
    if not code or not ex:
        return None
    return f"{ex}{code}"
