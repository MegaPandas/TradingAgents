"""Unit tests for the deterministic math self-check helpers.

These validators are pure-Python (no I/O, no LLM) so they can be exercised
exhaustively. They back the Neutral debator's probability/EV discipline and
the PM's FV-adjustment reconciliation.
"""

import pytest

from tradingagents.agents.utils.math_validators import (
    validate_expected_value,
    validate_fair_value_adjustment,
    validate_position_size_chain,
    validate_probability_sum,
    validate_vol_annualization,
)


@pytest.mark.unit
class TestProbabilitySum:
    def test_summing_to_one_passes(self):
        ok, msg = validate_probability_sum([0.30, 0.50, 0.20])
        assert ok is True
        assert "1.000" in msg

    def test_within_tolerance_passes(self):
        ok, _ = validate_probability_sum([0.33, 0.33, 0.34])
        assert ok is True

    def test_outside_tolerance_fails(self):
        ok, msg = validate_probability_sum([0.5, 0.6])
        assert ok is False
        assert "1.1000" in msg
        assert "Re-normalise" in msg

    def test_custom_tolerance(self):
        # 0.97 deviates 0.03 — fails default tol 0.01, passes tol 0.05.
        assert validate_probability_sum([0.97, 0.0])[0] is False
        assert validate_probability_sum([0.97, 0.0], tol=0.05)[0] is True

    def test_empty_fails(self):
        ok, msg = validate_probability_sum([])
        assert ok is False
        assert "no probabilities" in msg


@pytest.mark.unit
class TestExpectedValue:
    def test_weighted_sum(self):
        ok, msg = validate_expected_value([(0.30, 32.8), (0.50, 0.9), (0.20, -25.6)])
        assert ok is True
        # 0.3*32.8 + 0.5*0.9 + 0.2*(-25.6) = 9.84 + 0.45 - 5.12 = 5.17
        assert "E[R]=+5.17%" in msg

    def test_probabilities_must_sum_to_one(self):
        ok, msg = validate_expected_value([(0.5, 10.0), (0.6, -5.0)])
        assert ok is False
        assert "not 1.0" in msg

    def test_empty_fails(self):
        ok, msg = validate_expected_value([])
        assert ok is False
        assert "no scenarios" in msg


@pytest.mark.unit
class TestPositionSizeChain:
    def test_multiplicative_chain(self):
        # 30% base × 0.7 trend × 0.5 uncertainty = 10.5%
        ok, msg = validate_position_size_chain([30], [0.7, 0.5])
        assert ok is True
        assert "10.50%" in msg
        assert "0.7" in msg and "0.5" in msg

    def test_conservative_chain_bug_caught(self):
        # The audit case: 25% → ×0.7 = 17.5%, not the 13-15% the debator claimed.
        # The helper quotes the arithmetic so the discrepancy is visible.
        ok, msg = validate_position_size_chain([25], [0.7])
        assert ok is True
        assert "17.50%" in msg

    def test_empty_inputs_fail(self):
        assert validate_position_size_chain([], [0.5])[0] is False
        assert validate_position_size_chain([25], [])[0] is False


@pytest.mark.unit
class TestVolAnnualization:
    def test_daily_horizon_annual(self):
        # ATR=5, price=100, 21 trading days (monthly horizon).
        ok, msg = validate_vol_annualization(atr=5.0, price=100.0, periods=21)
        assert ok is True
        # daily = 0.05; horizon = 0.05 * sqrt(21) ≈ 0.22913; annual = 0.05*sqrt(252) ≈ 0.79373
        assert "daily=5.000%" in msg
        assert "horizon(21d)=22.913%" in msg
        assert "annual(252.0d)=79.373%" in msg

    def test_non_positive_rejected(self):
        assert validate_vol_annualization(5.0, 0.0, 21)[0] is False
        assert validate_vol_annualization(5.0, 100.0, 0)[0] is False
        assert validate_vol_annualization(5.0, -1.0, 21)[0] is False

    def test_custom_annualization_factor(self):
        ok, msg = validate_vol_annualization(2.0, 100.0, 63, annualization_factor=252.0)
        assert ok is True
        assert "daily=2.000%" in msg


@pytest.mark.unit
class TestFairValueAdjustment:
    def test_reconciling_adjustment_passes(self):
        # 85.08 × (1 + 0.029) × (1 - 0.005) = 85.08 × 1.024 ≈ 87.12
        ok, msg = validate_fair_value_adjustment(85.08, [+2.9, -0.5], 87.12)
        assert ok is True
        assert "computed=87.12" in msg or "computed=87.1" in msg

    def test_non_reconciling_adjustment_fails(self):
        ok, msg = validate_fair_value_adjustment(85.08, [+2.9, -0.5], 95.00)
        assert ok is False
        assert "Re-compute" in msg

    def test_non_positive_anchor_rejected(self):
        ok, msg = validate_fair_value_adjustment(0.0, [+2.0], 1.0)
        assert ok is False
        assert "anchor must be positive" in msg

    def test_empty_adjustments_passes_with_identity(self):
        # No adjustments → factor stays 1.0 → computed == anchor == adjusted.
        ok, msg = validate_fair_value_adjustment(100.0, [], 100.0)
        assert ok is True
        assert "factor=1.0000" in msg
