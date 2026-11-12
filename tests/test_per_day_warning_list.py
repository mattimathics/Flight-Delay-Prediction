"""Tests for src/per_day_warning_list.py."""

import sqlite3

import numpy as np
import pandas as pd

import model_v3
import per_day_warning_list as pdw


def test_whole_period_list_matches_the_published_rule():
    rng = np.random.default_rng(0)
    late = rng.integers(0, 2, 1000)
    risk = np.round(rng.random(1000), 2)  # rounded, so there are ties
    flags = pdw.whole_period_flags(risk)
    published = model_v3.flag_riskiest(late, risk)
    assert flags.sum() == published["flagged_flights"]
    assert np.isclose(late[flags].mean(), published["late_rate_in_flagged"])


def test_per_day_list_flags_one_in_ten_of_each_day():
    # Day 1 is stormy (every flight risky), day 2 calm: a whole-period list would take only day 1
    risk = np.r_[np.linspace(0.9, 0.99, 50), np.linspace(0.01, 0.1, 50)]
    day = np.r_[["2008-01-01"] * 50, ["2008-01-02"] * 50]
    flags = pdw.per_day_flags(risk, day)
    assert flags[:50].sum() == 5 and flags[50:].sum() == 5
    # The riskiest flights of each day are the ones flagged
    assert flags[45:50].all() and flags[95:100].all()
    whole = pdw.whole_period_flags(risk)
    assert whole[:50].sum() == 10 and whole[50:].sum() == 0


def test_score_counts_late_flights_on_the_list():
    late = np.array([1, 0, 1, 1, 0, 0])
    flagged = np.array([True, True, False, True, False, False])
    day = np.array(["a", "a", "a", "b", "b", "b"])
    s = pdw.score(late, flagged, day)
    assert s["flagged_flights"] == 3
    assert np.isclose(s["late_rate_on_list"], 2 / 3)
    assert np.isclose(s["share_of_late_flights_caught"], 2 / 3)
    assert s["days_with_flights_on_list"] == 2


def test_dates_are_looked_up_by_row_id(tmp_path):
    db = tmp_path / "f.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE flightdetails (Year INT, Month INT, DayofMonth INT)")
        con.executemany("INSERT INTO flightdetails VALUES (?, ?, ?)",
                        [(2007, 12, 31), (2008, 1, 2), (2008, 3, 15)])
    days = pdw.flight_dates(db, [3, 2])
    assert list(days.dt.strftime("%Y-%m-%d")) == ["2008-03-15", "2008-01-02"]


def test_compare_gives_both_lists_for_each_model():
    rng = np.random.default_rng(1)
    table = pd.DataFrame({"late": rng.integers(0, 2, 200), "2 hours before": rng.random(200)})
    day = np.repeat(["d1", "d2", "d3", "d4"], 50)
    result = pdw.compare(table, day)
    assert list(result["warning_list"]) == ["Whole Jan-Apr at once (published)", "Fresh list each day"]
    assert (result["flagged_flights"] == 20).all()
