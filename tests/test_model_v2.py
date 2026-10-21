"""Tests for src/model_v2.py on a tiny synthetic database (no real data needed)."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import model_v2

POST_DEPARTURE = {"DepTime", "DepDelay", "TaxiOut", "TaxiIn", "AirTime", "ArrTime", "ArrDelay",
                  "ActualElapsedTime", "CarrierDelay", "WeatherDelay", "NASDelay",
                  "SecurityDelay", "LateAircraftDelay", "Cancelled", "Diverted"}


@pytest.fixture
def db(tmp_path):
    """A small flightdetails/planes database shaped like the real Fly.db."""
    rng = np.random.default_rng(0)
    n = 2000
    flights = pd.DataFrame({
        "Year": rng.choice([2006, 2007, 2008], n),
        "Month": rng.integers(1, 13, n),
        "DayOfWeek": rng.integers(1, 8, n),
        "CRSDepTime": rng.integers(0, 2400, n),
        "CRSArrTime": rng.integers(0, 2400, n),
        "CRSElapsedTime": rng.integers(40, 300, n),
        "Distance": rng.integers(100, 2500, n),
        "UniqueCarrier": rng.choice(["AA", "UA", "WN"], n),
        "Origin": rng.choice(["ORD", "ATL", "SFO"], n),
        "Dest": rng.choice(["LAX", "JFK", "DEN"], n),
        "DepTime": rng.integers(0, 2400, n),
        "DepDelay": rng.integers(-10, 60, n),
        "TaxiOut": rng.integers(5, 30, n),
        "TaxiIn": rng.integers(2, 15, n),
        "AirTime": rng.integers(30, 280, n),
        "ArrDelay": rng.integers(-20, 90, n).astype(float),
        "TailNum": rng.choice(["N1", "N2", "N3", "N4"], n),
    })
    flights.loc[:9, "ArrDelay"] = np.nan  # cancelled flights have no arrival delay
    planes = pd.DataFrame({
        "tailnum": ["N1", "N2", "N3", "N4"],
        "year": ["1990", "2005", "None", "0000"],  # the real file has 'None' and '0000' years
    })
    path = tmp_path / "Fly.db"
    with sqlite3.connect(path) as conn:
        flights.to_sql("flightdetails", conn, index=False)
        planes.to_sql("planes", conn, index=False)
    return path


def test_no_post_departure_feature_in_pre_departure_lists():
    features = set(model_v2.PRE_DEPARTURE_NUM + model_v2.PRE_DEPARTURE_CAT)
    assert not features & POST_DEPARTURE


def test_leaky_list_is_post_departure_only():
    assert set(model_v2.LEAKY_NUM) <= POST_DEPARTURE


def test_delay_label_is_15_minutes_or_more(db):
    df = model_v2.load(db, pct=100)
    assert (df["Delayed"] == (df["ArrDelay"] >= 15).astype(int)).all()
    assert df.loc[df["ArrDelay"] == 15, "Delayed"].eq(1).all()
    assert df.loc[df["ArrDelay"] == 14, "Delayed"].eq(0).all()


def test_cancelled_flights_are_dropped(db):
    assert model_v2.load(db, pct=100)["ArrDelay"].notna().all()
    assert len(model_v2.load(db, pct=100)) == 1990


def test_implausible_plane_ages_become_missing(db):
    df = model_v2.load(db, pct=100)
    # 'None' and '0000' manufacture years would give ages of ~2000 years
    assert df.loc[df["PlaneAge"].notna(), "PlaneAge"].between(0, 60).all()
    assert df["PlaneAge"].isna().any()


@pytest.mark.parametrize("hhmm, expected", [
    (0, "Night"), (459, "Night"), (500, "Morning"), (1159, "Morning"), (1200, "Afternoon"),
    (1659, "Afternoon"), (1700, "Evening"), (2059, "Evening"), (2100, "Night"), (2359, "Night"),
])
def test_time_of_day_buckets(hhmm, expected):
    assert model_v2.time_of_day(pd.Series([hhmm]))[0] == expected


def test_sample_is_repeatable(db):
    a = model_v2.load(db, pct=20, seed=1)
    b = model_v2.load(db, pct=20, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_sample(db):
    a = model_v2.load(db, pct=20, seed=1)
    b = model_v2.load(db, pct=20, seed=2)
    assert not a.equals(b)


def test_sample_size_is_close_to_pct():
    kept = sum(model_v2.in_sample(rowid, pct=10, seed=42) for rowid in range(1, 100_001))
    assert 9_500 < kept < 10_500
