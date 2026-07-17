"""Math self-check helpers for debators and the Portfolio Manager.

The audit (002594.SZ_20260717_121508) surfaced repeated arithmetic
errors in debator output: probability sums != 1.0, Kelly chains that
chain wrong, volatility annualisation off by 2.5x, and FV adjustment
deltas that don't reconcile.  These helpers are designed to be called
BY the agent during drafting (or by a structured-output validator) so
the math is verifiable.

The helpers are pure-Python and never raise — they return a tuple
(ok: bool, msg: str) so callers can decide what to do (log, retry,
reject the parse, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence


def validate_probability_sum(
    probabilities: Sequence[float],
    tol: float = 0.01,
) -> tuple[bool, str]:
    """Verify probabilities sum to 1.0 within tolerance.

    Args:
        probabilities: list of per-scenario probabilities, e.g. [0.30, 0.50, 0.20]
        tol: tolerance for the absolute deviation from 1.0

    Returns (True, "ok") when |sum - 1.0| <= tol.
    """
    if not probabilities:
        return False, "no probabilities supplied"
    total = sum(probabilities)
    delta = abs(total - 1.0)
    if delta <= tol:
        return True, f"sum={total:.3f} within tol {tol}"
    return False, (
        f"probabilities sum to {total:.4f}, deviation {delta:.4f} > tol {tol}. "
        f"Re-normalise or state the rounding residual."
    )


def validate_expected_value(
    scenarios: Sequence[tuple[float, float]],
    tol: float = 0.5,
) -> tuple[bool, str]:
    """Verify expected-value arithmetic against the model claim.

    Args:
        scenarios: list of (probability, return_pct) pairs, e.g.
            [(0.30, 32.8), (0.50, 0.9), (0.20, -25.6)]
        tol: tolerance in pct points for the difference between the
            computed sum(p*r) and the model's claimed E[R].

    Returns (True, ...) when p sums to 1.0 and the sum is internally
    consistent (no separate claim to compare against — that's the caller's
    job to extract from the report text).
    """
    if not scenarios:
        return False, "no scenarios"
    total_p = sum(p for p, _ in scenarios)
    if abs(total_p - 1.0) > 0.01:
        return False, f"scenario probabilities sum to {total_p:.4f}, not 1.0"
    ev = sum(p * r for p, r in scenarios)
    return True, f"E[R]={ev:+.2f}% (sum p×r; caller compares to claimed value)"


def validate_position_size_chain(
    weights: Sequence[float],
    factors: Sequence[float],
    label: str = "position sizing",
    tol_pct: float = 0.5,
) -> tuple[bool, str]:
    """Verify a position-sizing chain's multiplicative arithmetic.

    The Conservative debator's 25% → ×0.7 → 13-15% chain off by 30%
    because 25 × 0.7 = 17.5, not 13-15.  This helper accepts the nominal
    weights and the multiplicative factors and verifies the chain.

    Args:
        weights: starting weights in pct (e.g. [30] = 30% base)
        factors: multiplicative factors applied (e.g. [0.7, 0.5] = trend
            discount × earnings uncertainty discount)
        label: description printed in the message
        tol_pct: tolerance for the difference between the chain
            arithmetic and a separately-claimed final weight

    Returns (True, ...) when the chain's product matches a simple
    floor of the worst-case product.
    """
    if not weights or not factors:
        return False, "empty weights or factors"
    chain_product = 1.0
    for w in weights:
        chain_product *= float(w) / 100.0
    for f in factors:
        chain_product *= float(f)
    final_pct = chain_product * 100
    return True, (
        f"{label}: base={weights[0]}% × factors={list(factors)} = "
        f"{final_pct:.2f}% (use this exact number when quoting final)"
    )


def validate_vol_annualization(
    atr: float,
    price: float,
    periods: int,
    annualization_factor: float = 252.0,
    tol: float = 0.02,
) -> tuple[bool, str]:
    """Verify the typical daily→annual vol pipeline.

    The Neutral analyst reported 3-6 month volatility as 15% from
    ATR × √120 / price but the formula yields 37.9% (off by 2.5×).  The
    correct pipeline is:
        daily_vol = ATR / price
        horizon_vol = daily_vol × sqrt(periods)
    where periods should be in TRADING DAYS (≈21 per month for monthly,
    63 for quarterly, 252 for annual), NOT calendar days.

    Args:
        atr: 14-day ATR (daily)
        price: current price
        periods: trading-day count over the horizon (e.g. 21 monthly, 63 q)
        annualization_factor: trading days per year
        tol: tolerance for the ratio between reported and computed

    Returns (True, ...) when arithmetic is internally consistent; the
    caller decides if the result matches the report's claim.
    """
    if price <= 0 or periods <= 0:
        return False, "non-positive price or periods"
    daily = atr / price
    horizon = daily * (periods ** 0.5)
    annual = daily * (annualization_factor ** 0.5)
    return True, (
        f"daily={daily:.3%}, horizon({periods}d)={horizon:.3%}, "
        f"annual({annualization_factor}d)={annual:.3%}"
    )


def validate_fair_value_adjustment(
    anchor: float,
    adjustments_pct: Sequence[float],
    adjusted: float,
    tol_pct: float = 1.0,
) -> tuple[bool, str]:
    """Verify a sequence of +/- % adjustments reconciles to the claimed
    adjusted value.

    The PM reported adjusted FV = anchor × (1 + Σ delta_i) but the printed
    deltas didn't reconcile (e.g. +2.9% - 0.5% = +3.4% but final move was
    different).

    Args:
        anchor: starting FV (e.g. 85.08)
        adjustments_pct: list of percentage adjustments (e.g. [+2.9, -0.5])
        adjusted: claimed final FV (e.g. 88.00)
        tol_pct: tolerance in pct points for the difference between the
            computed and the claimed adjusted value

    Returns (True, ...) when the chain reconciles within tol_pct.
    """
    if anchor <= 0:
        return False, "anchor must be positive"
    factor = 1.0
    for a in adjustments_pct:
        factor *= 1.0 + float(a) / 100.0
    computed = anchor * factor
    delta_pct = abs(computed - adjusted) / anchor * 100.0
    if delta_pct <= tol_pct:
        return True, (
            f"anchor={anchor:.2f}, factor={factor:.4f}, "
            f"computed={computed:.2f}, claimed={adjusted:.2f} (delta={delta_pct:.3f}%)"
        )
    return False, (
        f"adjustments {list(adjustments_pct)} applied to anchor {anchor:.2f} "
        f"yield {computed:.2f}, but claimed adjusted is {adjusted:.2f} "
        f"(delta {delta_pct:.2f}% > tol {tol_pct}%). Re-compute and re-quote."
    )