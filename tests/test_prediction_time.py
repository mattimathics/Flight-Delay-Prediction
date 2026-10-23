"""Tests for src/prediction_time.py on the random database in conftest.py."""

import sqlite3

import model_v3
import prediction_time


def test_one_row_per_prediction_time(db):
    import matplotlib
    matplotlib.use("Agg")
    # The random database never lands a plane where it next departs from; make some do
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET Dest = 'ORD' WHERE rowid % 3 = 0")
    table = prediction_time.compare_prediction_times(model_v3.load(db, pct=100, seed=42), db)
    assert list(table["prediction_time"]) == ["Weeks ahead (timetable only)", "4 hours before",
                                              "2 hours before", "1 hour before", "After take-off (leaky)"]
    assert table["test_auc"].between(0, 1).all()
    ax = prediction_time.plot_prediction_time(table)
    assert len(ax.get_lines()) == 3  # honest scores, leaky line, coin-flip line
