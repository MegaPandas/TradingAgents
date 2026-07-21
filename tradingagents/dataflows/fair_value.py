"""Deterministic fair-value calculator — 5 academic models + turnaround path.

All inputs come from akshare financial data (TTM EPS, BVPS, annualized
ROE, revenue growth). No LLM judgment, no analyst estimates — same
inputs always produce the same output. Returns a markdown table of 5
model results + a fair-value range (min / median / max).

Models (standard — require positive EPS):
1. Gordon Growth:    FV = TTM_EPS × (1+g) / (r - g)  [g = min(revenue_growth, 6%)]
2. Graham:           FV = TTM_EPS × (8.5 + 2×max(0,g%)) × bond_yield_adj
3. Residual Income:  FV = BVPS + (ROE - r) × BVPS / (r - g)
4. PEG (PEG=1.0):    FV = growth% × TTM_EPS  (only when growth >= 12%)
5. PB Normalization: FV = justified_PB × BVPS  [justified_PB = (ROE-g)/(r-g)]

Turnaround path (activated when TTM EPS ≤ 0 or < 2 models valid):
A. P/B vs 5-year historical range: FV = percentile_PB × BVPS
B. Asset-based floor:              FV = BVPS × 0.5 (going-concern floor)

References: Damodaran (NYU Stern), Graham (Intelligent Investor),
Ohlson (1995), CFA Institute curriculum.
"""
from __future__ import annotations

import logging
import re
import statistics
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# Per-code cache for 5-year PB history (baidu valuation endpoint).
# Cached separately from the financial-abstract cache in akshare_data
# because the data source is independent.
_PB_HISTORY_CACHE: dict[str, pd.DataFrame] = {}

_DEFAULTS = {
    "discount_rate": 0.10,
    "terminal_growth_floor": 0.03,
    "terminal_growth_cap": 0.06,
    "peg_min_growth": 0.12,     # PEG model only emits when growth >= 12%
    "bond_yield": 0.025,        # CN 10y treasury proxy for Graham adjustment
    "graham_base_pe": 4.4,      # Graham's original base PE for zero-growth at 4.4% bond yield
}

_CURRENCY = "CNY"


def _safe_float(v) -> float | None:
    """Coerce a value to float, returning None on a non-numeric input.

    Shared by ``_compute_cagr`` and ``_extract_financial_metrics._val`` so the
    eastmoney cells (strings, None, placeholder text) parse through one path.
    Does NOT handle NaN — callers that need NaN-rejection (``_val``) check
    ``pd.isna`` first.
    """
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _annualize_roe(roe_val: float, period_col: str) -> float:
    """Annualize a YTD ROE based on the period's month.

    Q1 (0331) → ×4, H1 (0630) → ×2, 9M (0930) → ×12/9, Annual (1231) → ×1.
    """
    if len(period_col) < 8:
        return roe_val
    mm = period_col[4:6]
    annualization = {
        "03": 4.0, "06": 2.0, "09": 12.0 / 9.0, "12": 1.0,
    }.get(mm, 1.0)
    return roe_val * annualization


def _compute_cagr(abstract: pd.DataFrame, keyword: str) -> float | None:
    """Compute 3yr revenue CAGR from annual columns if available."""
    annual_cols = sorted(
        [c for c in abstract.columns if c not in ("选项", "指标") and c.endswith("1231")],
        reverse=True
    )
    if len(annual_cols) < 4:
        return None
    rows = abstract[abstract["指标"].astype(str).str.contains(re.escape(keyword), na=False)]
    if rows.empty:
        return None
    latest = rows[annual_cols[0]].iloc[0]
    oldest = rows[annual_cols[3]].iloc[0]  # 3 years back
    latest_f = _safe_float(latest)
    oldest_f = _safe_float(oldest)
    if latest_f is None or oldest_f is None:
        return None
    if oldest_f <= 0 or latest_f <= 0:
        return None
    cagr = (latest_f / oldest_f) ** (1.0 / 3.0) - 1.0
    return min(max(cagr, 0.0), 0.50)


def _extract_financial_metrics(abstract: pd.DataFrame) -> dict:
    """Extract TTM EPS + annualized metrics from stock_financial_abstract."""
    if abstract is None or abstract.empty:
        return {}

    # Sort period columns descending (newest first)
    period_cols = sorted(
        [c for c in abstract.columns if c not in ("选项", "指标")],
        reverse=True,
    )
    if not period_cols:
        return {}

    def _find_rows(keyword: str) -> pd.DataFrame:
        mask = abstract["指标"].astype(str).str.contains(re.escape(keyword), na=False)
        return abstract[mask]

    def _val(df: pd.DataFrame, col: str) -> float | None:
        if df.empty or col not in df.columns:
            return None
        v = df[col].iloc[0]
        if pd.isna(v):
            return None
        return _safe_float(v)

    eps_rows = _find_rows("稀释每股收益")
    if eps_rows.empty:
        eps_rows = _find_rows("基本每股收益")

    bvps_rows = _find_rows("每股净资产")
    roe_rows = _find_rows("净资产收益率")
    if roe_rows.empty:
        roe_rows = _find_rows("ROE")
    growth_rows = _find_rows("营业总收入增长率")
    if growth_rows.empty:
        growth_rows = _find_rows("营业收入增长率")

    annual_cols = [c for c in period_cols if c.endswith("1231")]
    latest_col = period_cols[0]

    # --- TTM EPS ---
    if annual_cols and latest_col != annual_cols[0] and not latest_col.endswith("1231"):
        fy_col = annual_cols[0]
        fy_eps = _val(eps_rows, fy_col)
        q_eps = _val(eps_rows, latest_col)
        # prior-year same quarter
        if len(latest_col) == 8:
            yyyy = latest_col[:4]
            mmdd = latest_col[4:]
            prior_col = str(int(yyyy) - 1) + mmdd
            prior_eps = _val(eps_rows, prior_col) if prior_col in period_cols else None
        else:
            prior_eps = None
        if fy_eps is not None and q_eps is not None and prior_eps is not None:
            ttm_eps = fy_eps + q_eps - prior_eps
        elif fy_eps is not None:
            ttm_eps = fy_eps
        else:
            ttm_eps = q_eps
    elif annual_cols:
        ttm_eps = _val(eps_rows, annual_cols[0])
    else:
        ttm_eps = _val(eps_rows, latest_col)

    if ttm_eps is None:
        return {}

    # --- ROE annualized ---
    ref_col = annual_cols[0] if annual_cols else latest_col
    roe_raw = _val(roe_rows, ref_col)
    if roe_raw is not None and ref_col.endswith("1231"):
        roe_pct = roe_raw  # already annual
    elif roe_raw is not None:
        roe_pct = _annualize_roe(roe_raw, ref_col)
    else:
        roe_pct = None

    bvps = _val(bvps_rows, ref_col)
    revenue_growth_pct = _val(growth_rows, ref_col)

    # --- 3yr CAGR ---
    cagr = _compute_cagr(abstract, "营业总收入")

    # --- FY EPS (latest annual) for cyclical-trough detection ---
    fy_eps = None
    if annual_cols:
        fy_eps = _val(eps_rows, annual_cols[0])

    return {
        "ttm_eps": ttm_eps,
        "fy_eps": fy_eps,
        "bvps": bvps,
        "roe_pct": roe_pct,
        "revenue_growth_pct": revenue_growth_pct,
        "cagr_3yr": cagr,
        "period_date": latest_col,
        "annual_date": ref_col,
    }


def _pb_history(code: str) -> pd.DataFrame | None:
    """Fetch 5-year daily P/B history for a CN stock (cached per code).

    Uses akshare's ``stock_zh_valuation_baidu`` endpoint — the same
    underlying source that powers the Baidu stock page valuation charts.
    Only ``市净率`` (P/B) is reliably available; 市盈率 / 市销率 return
    NoneType for many tickers.

    Returns None on any failure so callers degrade gracefully.
    """
    cached = _PB_HISTORY_CACHE.get(code)
    if cached is not None:
        return cached
    try:
        from tradingagents.dataflows.akshare_data import _require_akshare, _no_proxy

        ak = _require_akshare()
        with _no_proxy():
            df = ak.stock_zh_valuation_baidu(symbol=code, indicator="市净率")
        if df is not None and not df.empty and "value" in df.columns:
            _PB_HISTORY_CACHE[code] = df
            return df
    except Exception as exc:
        logger.warning("PB history fetch failed for %s: %s", code, exc)
    return None


def _turnaround_valuation(code: str, m: dict, params: dict) -> str:
    """Valuation for negative-EPS / negative-ROE companies.

    Activated when the 5 standard academic models produce < 2 valid
    outputs (typically because TTM EPS ≤ 0 or ROE is deeply negative).
    Uses methods that do NOT depend on positive earnings:

    A. **P/B vs 5-year historical range** — assumes the market's
       historical P/B multiple for this stock encodes its normalised
       earnings power, brand value, and asset productivity.  The median
       historical P/B × current BVPS gives a "regression-to-mean" anchor.

    B. **Asset-based floor** — BVPS × a conservative going-concern
       discount.  Not a liquidation value; rather a "what a strategic
       buyer would pay for the tangible net assets" floor.

    Returns a markdown string appended to the main fair-value report.
    """
    pb_df = _pb_history(code)
    bvps = m.get("bvps")
    ttm_eps = m.get("ttm_eps", 0)

    lines: list[str] = [
        "",
        "## Turnaround Valuation",
        "",
        "> ⚠ **Standard models are inapplicable**: TTM EPS is negative or near-zero, "
        "and/or ROE is deeply negative, causing 4+ of the 5 academic models to "
        "return N/A. The anchors below use balance-sheet and market-history methods "
        "that do not require positive earnings. **These are directional guides, not "
        "precision targets.** The true value depends on whether and when the company "
        "returns to sustainable profitability.",
        "",
    ]

    methods: list[tuple[str, float | None, str]] = []

    # --- Method A: P/B vs 5-year historical range ---
    if pb_df is not None and bvps is not None and bvps > 0:
        pb_values = pb_df["value"].dropna()
        if len(pb_values) >= 60:  # at least ~3 months of daily data
            pb_p10 = float(pb_values.quantile(0.10))
            pb_p25 = float(pb_values.quantile(0.25))
            pb_p50 = float(pb_values.quantile(0.50))
            pb_p75 = float(pb_values.quantile(0.75))
            pb_p90 = float(pb_values.quantile(0.90))
            pb_low = float(pb_values.min())
            pb_high = float(pb_values.max())
            pb_latest = float(pb_values.iloc[-1])

            fv_p10 = round(pb_p10 * bvps, 2)
            fv_p25 = round(pb_p25 * bvps, 2)
            fv_p50 = round(pb_p50 * bvps, 2)
            fv_p75 = round(pb_p75 * bvps, 2)
            fv_p90 = round(pb_p90 * bvps, 2)
            fv_low = round(pb_low * bvps, 2)
            fv_high = round(pb_high * bvps, 2)
            fv_latest = round(pb_latest * bvps, 2)

            data_points = len(pb_values)
            date_span = f"{pb_df['date'].iloc[0]} → {pb_df['date'].iloc[-1]}"

            lines += [
                "### A. P/B vs Historical Range",
                "",
                f"- Current BVPS: **{bvps:.2f}**",
                f"- Historical window: {date_span} ({data_points} daily observations)",
                f"- Latest P/B: **{pb_latest:.2f}×** → implied FV: **{fv_latest:.2f}**",
                "",
                "| Percentile | P/B (×) | Fair Value (CNY) |",
                "|---|---:|---:|",
                f"| Minimum | {pb_low:.2f} | {fv_low:.2f} |",
                f"| P10 | {pb_p10:.2f} | {fv_p10:.2f} |",
                f"| P25 | {pb_p25:.2f} | {fv_p25:.2f} |",
                f"| **P50 (median)** | **{pb_p50:.2f}** | **{fv_p50:.2f}** |",
                f"| P75 | {pb_p75:.2f} | {fv_p75:.2f} |",
                f"| P90 | {pb_p90:.2f} | {fv_p90:.2f} |",
                f"| Maximum | {pb_high:.2f} | {fv_high:.2f} |",
                "",
            ]

            # Median PB → primary turnaround anchor
            methods.append((
                "P/B Median (historical)",
                fv_p50,
                f"Median PB {pb_p50:.2f}× × BVPS {bvps:.2f}",
            ))
            # P25 PB → conservative anchor for risk-averse sizing
            methods.append((
                "P/B P25 (historical)",
                fv_p25,
                f"P25 PB {pb_p25:.2f}× × BVPS {bvps:.2f}",
            ))
        else:
            lines += [
                "### A. P/B vs Historical Range",
                "",
                f"> Insufficient data: only {len(pb_values)} daily observations "
                "(need ≥ 60). Skipping P/B anchor.",
                "",
            ]
    else:
        lines += [
            "### A. P/B vs Historical Range",
            "",
            "> P/B history unavailable — baidu valuation endpoint returned no data.",
            "",
        ]

    # --- Method B: Asset-based floor ---
    if bvps is not None and bvps > 0:
        fv_asset_floor = round(bvps * 0.5, 2)
        lines += [
            "### B. Asset-Based Floor",
            "",
            f"- BVPS: **{bvps:.2f}**",
            f"- Conservative going-concern floor (0.5× BVPS): **{fv_asset_floor:.2f}**",
            "",
            "> Rationale: a strategic buyer would typically pay at least 0.5× "
            "tangible book for a going concern with an established brand, "
            "store network, or customer base — even if current earnings are negative.",
            "",
        ]
        methods.append((
            "Asset Floor (0.5× BVPS)",
            fv_asset_floor,
            f"BVPS {bvps:.2f} × 0.5",
        ))

    # --- Aggregate ---
    valid = [(n, fv, note) for n, fv, note in methods if fv is not None and fv > 0]
    if valid:
        fvs = [fv for _, fv, _ in valid]
        fv_min = min(fvs)
        fv_max = max(fvs)
        fv_median = statistics.median(fvs) if len(fvs) >= 3 else (
            statistics.median(fvs) if len(fvs) == 2 else fvs[0]
        )

        lines += [
            "### Turnaround Fair Value Summary",
            "",
            "| Method | Fair Value (CNY) | Formula |",
            "|---|---:|---|",
        ]
        for name, fv, note in valid:
            lines.append(f"| {name} | {fv:.2f} | {note} |")

        lines += [
            "",
            f"- **Low**: {fv_min:.2f}",
            f"- **Median**: {fv_median:.2f}",
            f"- **High**: {fv_max:.2f}",
            f"- Valid turnaround methods: {len(valid)}",
            "",
            "> 💡 **Use these as valuation bounds, not precise targets.** ",
            "> If a recent transaction price exists (strategic investment, M&A, ",
            "> block trade), it provides a stronger anchor than any model-based ",
            "> estimate for a turnaround situation. Cite it in the Fundamentals ",
            "> report as the primary reference point.",
            "",
        ]

    return "\n".join(lines)


def _parse_turnaround_fvs(turnaround_block: str) -> list[float]:
    """Extract fair-value numbers from the Turnaround Fair Value Summary table.

    Only parses rows in the summary section (not the raw PB percentile
    table, where column 2 is the P/B multiple, not an FV).  Matches
    rows like ``| Method Name | 2.70 | formula |``.
    """
    fvs: list[float] = []
    if not turnaround_block:
        return fvs
    in_summary = False
    for line in turnaround_block.split("\n"):
        stripped = line.strip()
        if stripped.startswith("### Turnaround Fair Value Summary"):
            in_summary = True
            continue
        if not in_summary:
            continue
        # Section break: next heading or blank line after the table ends
        if stripped.startswith("##") or stripped.startswith("> "):
            break
        parts = [p.strip() for p in stripped.split("|")]
        # Summary table rows: | Method Name | FV | Formula |
        # → split yields ['', 'Method Name', 'FV', 'Formula', '']
        if len(parts) >= 4:
            try:
                fv = float(parts[2])
                if fv > 0:
                    fvs.append(fv)
            except (ValueError, TypeError):
                pass
    return fvs


def calculate_fair_value(
    ticker: str,
    params: dict | None = None,
) -> str:
    """Compute fair value using 5 deterministic academic models."""
    from tradingagents.dataflows.akshare_data import _financial_abstract, _guard_cn

    code = _guard_cn(ticker)
    p = {**_DEFAULTS, **(params or {})}
    r = p["discount_rate"]
    g_floor = p["terminal_growth_floor"]
    g_cap = p["terminal_growth_cap"]
    peg_min_g = p["peg_min_growth"]
    bond_yield = p["bond_yield"]
    graham_base = p["graham_base_pe"]

    abstract = _financial_abstract(code)
    m = _extract_financial_metrics(abstract)

    if not m or m.get("ttm_eps") is None:
        return (
            f"# Fair Value Calculation for {code}\n"
            f"# Unable to calculate — missing financial data (EPS not found).\n"
        )

    eps = m["ttm_eps"]
    bvps = m.get("bvps")
    roe_pct = m.get("roe_pct")
    revenue_growth_pct = m.get("revenue_growth_pct")
    cagr = m.get("cagr_3yr")
    fy_eps = m.get("fy_eps")

    # Auto-detect cyclical trough: if TTM EPS < 85% of FY EPS, the TTM is
    # depressed by a one-off quarter(s).  Use FY EPS as normalized anchor so
    # the academic models don't understate fair value.  Caller-supplied
    # normalized_eps (in params) overrides the auto-detected value.
    normalized_eps = p.get("normalized_eps")
    if normalized_eps is None and fy_eps is not None and fy_eps > 0 and eps > 0:
        if (eps / fy_eps) < 0.85:
            normalized_eps = fy_eps

    # Choose growth rate: prefer 3yr CAGR (more stable), fall back to YoY
    if cagr is not None:
        g_display = cagr * 100
        g_source = "3yr CAGR"
    elif revenue_growth_pct is not None:
        g_display = revenue_growth_pct
        g_source = "YoY"
    else:
        g_display = g_floor * 100
        g_source = "default"

    # Terminal growth: use actual growth capped, not hardcoded 3%
    g_rev = min(max(g_display / 100.0, g_floor), g_cap)

    # Graham bond-yield adjustment factor (4.4 / current bond yield)
    graham_adj = 4.4 / (bond_yield * 100) if bond_yield > 0 else 1.0

    models: list[tuple[str, float | None, str]] = []

    # --- Model 1: Gordon Growth ---
    if eps > 0 and r > g_rev:
        fv_gordon = eps * (1 + g_rev) / (r - g_rev)
        models.append((
            "Gordon Growth",
            fv_gordon,
            f"TTM EPS {eps:.2f} × (1+{g_rev:.1%}) / ({r:.0%}-{g_rev:.1%})  [g={g_source}]"
        ))
    elif eps <= 0:
        models.append(("Gordon Growth", None, f"TTM EPS {eps:.2f} ≤ 0 — N/A"))
    else:
        models.append(("Gordon Growth", None, f"r ({r:.0%}) ≤ g ({g_rev:.1%})"))

    # --- Model 2: Graham (with bond-yield adjustment) ---
    if eps > 0:
        g_growth_floor = max(0.0, g_display)
        graham_pe = (graham_base + 2 * g_growth_floor) * graham_adj
        fv_graham = eps * graham_pe
        models.append((
            "Graham (adj.)",
            fv_graham,
            f"TTM EPS {eps:.2f} × PE {graham_pe:.1f} ({graham_base}+2×{g_growth_floor:.1f}%)×{graham_adj:.2f}"
        ))
    else:
        models.append((
            "Graham (adj.)",
            None,
            f"TTM EPS {eps:.2f} ≤ 0 — N/A (Graham requires positive earnings)"
        ))

    # --- Model 3: Residual Income (Ohlson) ---
    if bvps is not None and roe_pct is not None and r > g_rev:
        roe_dec = roe_pct / 100.0
        ri_spread = roe_dec - r
        fv_ri = bvps + (ri_spread * bvps) / (r - g_rev)
        models.append((
            "Residual Income",
            fv_ri,
            f"BVPS {bvps:.2f} + (ROE {roe_pct:.1f}%-{r:.0%})×BVPS / ({r:.0%}-{g_rev:.1%})"
        ))
    else:
        models.append(("Residual Income", None, "missing BVPS or ROE"))

    # --- Model 4: PEG (only for genuine growth names, PEG=1.0) ---
    g_dec = g_display / 100.0
    if g_dec >= peg_min_g and eps > 0:
        peg_fair_pe = g_dec * 100
        fv_peg = peg_fair_pe * eps
        models.append((
            "PEG (PEG=1.0)",
            fv_peg,
            f"PE {peg_fair_pe:.1f} (=growth {g_dec:.1%}) × TTM EPS {eps:.2f}  [growth ≥ {peg_min_g:.0%}]"
        ))
    else:
        models.append(("PEG (PEG=1.0)", None, f"growth {g_dec:.1%} < {peg_min_g:.0%} threshold — N/A"))

    # --- Model 5: Justified PB ---
    if bvps is not None and roe_pct is not None and r > g_rev:
        roe_dec = roe_pct / 100.0
        justified_pb = (roe_dec - g_rev) / (r - g_rev)
        # Floor at 0.3× for deeply negative ROE (going-concern floor),
        # but still flag it as weak — a negative-ROE company trading at 3×
        # fallback is the OLD behaviour that produced garbage 0.10 CNY
        # outputs for turnaround names.
        justified_pb = max(justified_pb, 0.3)
        fv_pb = justified_pb * bvps
        weak_flag = " ⚠ weak anchor" if roe_pct < 0 else ""
        models.append((
            f"Justified PB{weak_flag}",
            fv_pb,
            f"PB {justified_pb:.2f}× (= (ROE {roe_pct:.1f}%-{g_rev:.1%})/({r:.0%}-{g_rev:.1%})) × BVPS {bvps:.2f}"
        ))
    elif bvps is not None:
        # BVPS exists but ROE missing or r ≤ g → can't compute justified PB.
        # Use a conservative 0.5× BVPS floor instead of the old 3× fallback
        # which was produce nonsense for companies with negative earnings.
        fv_pb = 0.5 * bvps
        models.append((
            "PB (0.5× BVPS fallback)",
            fv_pb,
            f"PB 0.5× × BVPS {bvps:.2f}  [justified PB N/A — missing ROE or r ≤ g]"
        ))
    else:
        models.append(("Justified PB", None, "missing BVPS"))

    # --- Aggregate (TTM anchor) ---
    valid_fvs = [fv for _, fv, _ in models if fv is not None and fv > 0]
    if valid_fvs:
        fv_min = min(valid_fvs)
        fv_max = max(valid_fvs)
        fv_median = statistics.median(valid_fvs)
    else:
        fv_min = fv_max = fv_median = None

    # --- Turnaround path: when standard models fail ---
    # Trigger when < 2 models produce a valid (positive) output (typically
    # because TTM EPS ≤ 0). The turnaround module uses P/B history and
    # asset-based methods that don't require positive earnings.
    turnaround_block = ""
    if len(valid_fvs) < 2:
        turnaround_block = _turnaround_valuation(code, m, p)
        # Replace weak standard-model anchors with turnaround methods.
        # The "⚠ weak anchor" models (Justified PB at 0.3× floor for
        # negative-ROE companies) produce economically meaningless outputs
        # (e.g. 0.06 CNY) and drag the aggregate toward garbage.
        _tfvs = _parse_turnaround_fvs(turnaround_block)
        if _tfvs:
            # Use turnaround FVs as the primary range; keep any
            # non-weak standard-model FVs as supplementary.
            strong_fvs = [
                fv for name, fv, _ in models
                if fv is not None and fv > 0 and "weak anchor" not in name
            ]
            valid_fvs = strong_fvs + _tfvs
        else:
            valid_fvs.extend(_tfvs)
        if valid_fvs:
            fv_min = min(valid_fvs)
            fv_max = max(valid_fvs)
            fv_median = statistics.median(valid_fvs)

    # --- Cyclical-trough detection ---
    # If TTM EPS is materially below the most-recent FY EPS, the TTM is
    # depressed by a one-off quarter(s).  The PM should anchor on the
    # normalized (FY-anchored) EPS in that case, not the TTM.
    # fy_eps comes from _extract_financial_metrics (single source of truth).
    trough_flag = ""
    if fy_eps is not None and fy_eps > 0 and eps > 0:
        trough_ratio = eps / fy_eps
        if trough_ratio < 0.85:
            trough_flag = (
                f"⚠ TTM EPS {eps:.2f} = {trough_ratio:.0%} of FY EPS {fy_eps:.2f}. "
                f"Cyclical trough detected — consider normalized_eps ≈ {fy_eps:.2f} (FY anchor) "
                f"or higher if recovery is visible in the next quarter's data."
            )

    # --- Normalized-EPS re-run (if caller supplied) ---
    lines_norm = []
    if normalized_eps is not None and normalized_eps > 0:
        models_norm: list[tuple[str, float | None, str]] = []
        if r > g_rev:
            fv = normalized_eps * (1 + g_rev) / (r - g_rev)
            models_norm.append((
                "Gordon Growth (norm.)", fv,
                f"norm. EPS {normalized_eps:.2f} × (1+{g_rev:.1%}) / ({r:.0%}-{g_rev:.1%})"
            ))
        g_growth_floor = max(0.0, g_display)
        graham_pe = (graham_base + 2 * g_growth_floor) * graham_adj
        models_norm.append((
            "Graham (norm.)",
            normalized_eps * graham_pe,
            f"norm. EPS × PE {graham_pe:.1f}",
        ))
        if bvps is not None and roe_pct is not None and r > g_rev:
            roe_dec = roe_pct / 100.0
            ri_spread = roe_dec - r
            fv = bvps + (ri_spread * bvps) / (r - g_rev)
            models_norm.append((
                "Residual Income (norm.)", fv,
                f"BVPS {bvps:.2f} + (ROE-r)×BVPS / (r-g)"
            ))
        if g_dec >= peg_min_g and normalized_eps > 0:
            models_norm.append((
                "PEG (norm.)",
                g_dec * 100 * normalized_eps,
                f"PE {g_dec*100:.1f} × norm. EPS {normalized_eps:.2f}"
            ))
        if bvps is not None and roe_pct is not None and r > g_rev:
            roe_dec = roe_pct / 100.0
            justified_pb = (roe_dec - g_rev) / (r - g_rev)
            justified_pb = max(justified_pb, 0.5)
            models_norm.append((
                "Justified PB (norm.)",
                justified_pb * bvps,
                f"PB {justified_pb:.2f}× × BVPS {bvps:.2f}"
            ))
        valid_norm = [fv for _, fv, _ in models_norm if fv is not None and fv > 0]
        if valid_norm:
            fv_norm_min = min(valid_norm)
            fv_norm_max = max(valid_norm)
            fv_norm_median = statistics.median(valid_norm)
        else:
            fv_norm_min = fv_norm_max = fv_norm_median = None

        lines_norm = [
            "",
            "## Normalized-EPS Fair Value (cyclical-adjustment anchor)",
            f"- Normalized EPS (caller-supplied): {normalized_eps:.2f}",
            f"- **Median**: {fv_norm_median:.2f}" if fv_norm_median else "",
            f"- **Low**: {fv_norm_min:.2f}",
            f"- **High**: {fv_norm_max:.2f}",
            "",
            "| Model | Fair Value (CNY) | Formula |",
            "|---|---:|---|",
        ]
        for name, fv, formula in models_norm:
            fv_str = f"{fv:.2f}" if fv is not None and fv > 0 else "N/A"
            lines_norm.append(f"| {name} | {fv_str} | {formula} |")

    lines = [
        f"# Fair Value Calculation for {code}",
        f"# Currency: {_CURRENCY} | Latest period: {m['period_date']} | "
        f"Annual ref: {m.get('annual_date','?')} | Computed: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Input Summary",
        f"- TTM EPS: {eps:.4f}",
        f"- FY EPS (latest annual): {f'{fy_eps:.2f}' if fy_eps is not None else 'N/A'}",
        f"- BVPS: {f'{bvps:.2f}' if bvps is not None else 'N/A'}",
        f"- ROE (annualized): {f'{roe_pct:.2f}%' if roe_pct is not None else 'N/A'}",
        f"- Revenue growth ({g_source}): {g_display:.2f}%",
        f"- 3yr CAGR: {f'{cagr:.2%}' if cagr is not None else 'N/A'}",
        f"- Terminal growth used (g): {g_rev:.1%}  [floor {g_floor:.0%}, cap {g_cap:.0%}]",
        f"- Discount rate (r): {r:.0%}",
        f"- Bond yield (Graham adj): {bond_yield:.1%}",
    ]
    if trough_flag:
        lines += ["", f"## Cyclical Trough Warning", trough_flag]
    lines += [
        "",
        "## Model Results (TTM anchor)",
        "| Model | Fair Value (CNY) | Formula |",
        "|---|---:|---|",
    ]

    for name, fv, formula in models:
        fv_str = f"{fv:.2f}" if fv is not None and fv > 0 else "N/A"
        lines.append(f"| {name} | {fv_str} | {formula} |")

    if fv_median is not None:
        lines += [
            "",
            "## Fair Value Range (TTM anchor)",
            f"- **Low**: {fv_min:.2f}",
            f"- **Median**: {fv_median:.2f}",
            f"- **High**: {fv_max:.2f}",
            f"- Valid models: {len(valid_fvs)}/{len(models)}",
        ]

    if turnaround_block:
        lines.append(turnaround_block)

    return "\n".join(lines) + "\n".join(lines_norm)
