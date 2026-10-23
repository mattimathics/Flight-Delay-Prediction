"""Tests for src/model_v4.py: a hand-made day at one airport, so every answer is known."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import model_v4
from test_model_v2 import POST_DEPARTURE

# Our flight: plane N1 leaves ORD at 18:00 on 10 Jan 2007, so the cutoff is 16:00.
# Columns: flight, tail, origin, dest, scheduled dep, scheduled arr, actual arr, flight minutes, dep delay
FLIGHTS = [
    ("ours", "N1", "ORD", "LAX", 1800, 2030, 2045, 255, 10),
    ("plane's previous", "N1", "DEN", "ORD", 1300, 1600, 1630, 150, 60),  # took off 14:00 ORD time
    ("too early", "N2", "ORD", "ATL", 1255, 1500, 1500, 120, 60),         # before the 5-hour window
    ("in window, late", "N3", "ORD", "ATL", 1300, 1500, 1520, 120, 20),
    ("in window, on time", "N4", "ORD", "ATL", 1400, 1600, 1600, 120, 0),
    ("last known, late", "N5", "ORD", "ATL", 1545, 1745, 1830, 120, 40),  # 15 min late is known at 16:00
    ("too late to know", "N6", "ORD", "ATL", 1550, 1750, 1900, 120, 60),
]


def make_db(path, flights=FLIGHTS):
    rows = pd.DataFrame(flights, columns=["Name", "TailNum", "Origin", "Dest", "CRSDepTime", "CRSArrTime",
                                          "ArrTime", "ActualElapsedTime", "DepDelay"])
    rows = rows.assign(Year=2007, Month=1, DayofMonth=10, DayOfWeek=4, CRSElapsedTime=120, Distance=800,
                       UniqueCarrier="AA", DepTime=rows["CRSDepTime"], TaxiOut=10, TaxiIn=5, AirTime=100,
                       ArrDelay=rows["DepDelay"].astype(float))
    with sqlite3.connect(path) as conn:
        rows.drop(columns="Name").to_sql("flightdetails", conn, index=False)
        pd.DataFrame({"tailnum": ["N1"], "year": ["2000"]}).to_sql("planes", conn, index=False)
    return path


def our_flight(path):
    df = model_v4.load(path, pct=100, seed=42)
    return df[(df["Origin"] == "ORD") & (df["CRSDepTime"] == 1800)].iloc[0]


@pytest.fixture
def day(tmp_path):
    return make_db(tmp_path / "day.db")


def test_no_post_departure_feature_in_any_setup():
    for _, num, cat in model_v4.SETUPS.values():
        assert not set(num + cat) & POST_DEPARTURE


def test_airport_share_uses_only_flights_known_by_the_cutoff(day):
    # In the window: 13:00 (late), 14:00 (on time), 15:45 (late) -> 2 of 3
    assert our_flight(day)["AirportLateShareEarlier"] == pytest.approx(2 / 3)


def test_turnaround_is_scheduled_gap(day):
    assert our_flight(day)["PlaneTurnaroundMinutes"] == 120  # landed 16:00, leaves 18:00


def test_previous_delay_is_used_when_it_took_off_before_the_cutoff(day):
    assert our_flight(day)["PlanePrevDepDelay"] == 60


def test_previous_delay_is_blank_when_it_took_off_after_the_cutoff(tmp_path):
    # Landed 17:30 after 60 minutes in the air: took off 16:30, after the 16:00 cutoff
    flights = [f if f[0] != "plane's previous" else ("plane's previous", "N1", "DEN", "ORD", 1300, 1600,
                                                      1730, 60, 60) for f in FLIGHTS]
    assert np.isnan(our_flight(make_db(tmp_path / "late.db", flights))["PlanePrevDepDelay"])


def test_nothing_after_the_cutoff_changes_the_features(tmp_path):
    before = our_flight(make_db(tmp_path / "a.db"))
    # Change what happens after 16:00: our own delay and the flight we could not know about yet
    changed = [(*f[:8], 999) if f[0] in ("ours", "too late to know") else f for f in FLIGHTS]
    after = our_flight(make_db(tmp_path / "b.db", changed))
    for col in model_v4.AIRPORT_EARLIER + model_v4.PLANE_PREVIOUS:
        assert before[col] == after[col]


def test_placeholder_tail_numbers_are_not_matched(tmp_path):
    flights = [(f[0], "0", *f[2:]) if f[0] in ("ours", "plane's previous") else f for f in FLIGHTS]
    assert np.isnan(our_flight(make_db(tmp_path / "junk.db", flights))["PlaneTurnaroundMinutes"])


def test_runs_end_to_end_on_the_random_database(db):
    import matplotlib
    matplotlib.use("Agg")
    # The random database never lands a plane where it next departs from; make some do
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET Dest = 'ORD' WHERE rowid % 3 = 0")
    train, test = model_v4.model_v2.split_by_year(model_v4.load(db, pct=100, seed=42))
    results, _ = model_v4.model_v3.fit_and_score(train, test, model_v4.SETUPS)
    assert list(results["model"])[1:] == list(model_v4.SETUPS)
    assert len(model_v4.plot_riskiest(results).patches) == 3
