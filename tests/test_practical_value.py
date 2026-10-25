"""Tests for src/practical_value.py on small made-up predictions and the tiny synthetic database."""

import matplotlib

matplotlib.use("Agg")

import pytest

import model_v2
import practical_value as pv


def test_flagging_table_on_a_known_example():
    # 10 flights, 4 late; the riskiest 2 are both late, the next 3 hold 1 more late flight
    late = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    risk = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    table = pv.flagging_table(late, risk, shares=[0.2, 0.5]).set_index("flag_riskiest_share")
    assert table.loc[0.2, "flagged_flights"] == 2
    assert table.loc[0.2, "late_among_flagged"] == 1.0
    assert table.loc[0.2, "share_of_late_flights_caught"] == 0.5
    assert table.loc[0.5, "late_among_flagged"] == 0.6
    assert table.loc[0.5, "share_of_late_flights_caught"] == 0.75


def test_flagging_more_flights_never_catches_fewer_late_ones():
    late = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    risk = [0.3, 0.9, 0.8, 0.1, 0.5, 0.6, 0.2, 0.4, 0.7, 0.0]
    caught = pv.flagging_table(late, risk, shares=[0.1, 0.3, 0.5, 1.0])["share_of_late_flights_caught"]
    assert caught.is_monotonic_increasing
    assert caught.iloc[-1] == 1.0


def test_a_perfect_ranking_has_average_precision_of_one():
    summary = pv.summary_table([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).iloc[0]
    assert summary["auc"] == 1.0
    assert summary["average_precision"] == pytest.approx(1.0)
    assert summary["share_late"] == 0.5


def test_runs_end_to_end_on_the_synthetic_database(db):
    df = model_v2.load(db, pct=100, seed=42)
    train, test = model_v2.split_by_year(df)
    risk = pv.predict_test_risk(train, test)
    assert len(risk) == len(test)
    assert ((risk >= 0) & (risk <= 1)).all()
    ax = pv.plot_precision_recall(test["Delayed"].values, risk)
    assert ax.get_xlabel().startswith("Share of all late flights caught")
