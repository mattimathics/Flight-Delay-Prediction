"""Tests for src/time_of_day.py on a tiny hand-made database (no real data needed)."""

import sqlite3

import pandas as pd
import pytest

import time_of_day


@pytest.fixture
def db(tmp_path):
    """Six flights whose slots and delays are easy to check by hand."""
    flights = pd.DataFrame({
        # An evening flight that left late at 21:30 is the case the chart is about.
        "CRSDepTime": [800, 1400, 1800, 1900, 2300, 2200],
        "DepTime":    [805, 1400, 1800, 2130, 2300, 2200],
        "ArrDelay":   [0.0, 20.0, 0.0, 150.0, 0.0, None],  # last flight never arrived
    })
    path = tmp_path / "Fly.db"
    with sqlite3.connect(path) as conn:
        flights.to_sql("flightdetails", conn, index=False)
    return path


def test_slots_in_order_of_the_day(db):
    table = time_of_day.delay_rates(db)
    assert list(table["Slot"]) == ["Morning", "Afternoon", "Evening", "Night"]


def test_late_evening_flight_counts_as_evening_when_grouped_by_schedule(db):
    table = time_of_day.delay_rates(db).set_index("Slot")
    assert table.loc["Evening", "RateScheduled"] == 0.5
    assert table.loc["Night", "RateScheduled"] == 0.0
    assert table.loc["Night", "RateActual"] == 0.5


def test_flights_that_never_arrived_are_left_out(db):
    table = time_of_day.delay_rates(db)
    assert table["FlightsScheduled"].sum() == 5


def test_chart_is_saved(db, tmp_path):
    out = tmp_path / "chart.png"
    time_of_day.draw_chart(time_of_day.delay_rates(db), out)
    assert out.stat().st_size > 0
