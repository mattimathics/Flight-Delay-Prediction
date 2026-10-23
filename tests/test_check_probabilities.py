"""Tests for src/check_probabilities.py on small made-up predictions."""

import numpy as np
import pytest

import check_probabilities as cp


def test_a_trustworthy_model_has_no_gap():
    # 10 flights predicted at 20%, 2 of them late; 10 at 80%, 8 of them late
    risk = [0.2] * 10 + [0.8] * 10
    late = [1, 1] + [0] * 8 + [1] * 8 + [0, 0]
    table = cp.calibration_table(late, risk)
    assert list(table["flights"]) == [10, 10]
    assert list(table["actually_late"]) == [0.2, 0.8]
    assert cp.average_gap(table) == pytest.approx(0)


def test_an_overconfident_model_has_a_gap():
    # Predicts 80% but only half are late: 30 points off
    table = cp.calibration_table([1, 0, 1, 0], [0.8] * 4)
    assert cp.average_gap(table) == pytest.approx(0.3)


def test_threshold_table_on_a_known_example():
    late = [1, 1, 0, 0]
    risk = [0.9, 0.4, 0.6, 0.1]
    row = cp.threshold_table(late, risk, thresholds=[0.5]).iloc[0]
    assert row["share_of_flights_flagged"] == 0.5    # 0.9 and 0.6
    assert row["late_among_flagged"] == 0.5          # only 0.9 was late
    assert row["share_of_late_flights_caught"] == 0.5


def test_no_flights_flagged_gives_blank_not_an_error():
    row = cp.threshold_table([1, 0], [0.1, 0.2], thresholds=[0.9]).iloc[0]
    assert row["share_of_flights_flagged"] == 0
    assert np.isnan(row["late_among_flagged"])


def test_alert_level_is_cost_ratio():
    # Acting costs 1, a missed delay costs 4: act once the chance is 25% or more
    assert cp.alert_level(1, 4) == 0.25


def test_calibration_chart_has_diagonal_and_model_line():
    import matplotlib
    matplotlib.use("Agg")
    table = cp.calibration_table([1, 0, 1, 0], [0.2, 0.2, 0.7, 0.7])
    assert len(cp.plot_calibration(table).get_lines()) == 2
