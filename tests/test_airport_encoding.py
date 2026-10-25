"""Tests for src/airport_encoding.py on the tiny synthetic database in conftest.py."""

import sqlite3

import pandas as pd

import airport_encoding
from test_model_v2 import POST_DEPARTURE


def test_no_post_departure_feature_in_any_setup():
    for _, num, cat in airport_encoding.SETUPS.values():
        assert not set(num + cat) & POST_DEPARTURE


def test_rate_setups_drop_the_airport_categories():
    # Each airport is either a category or a rate, never both, so the comparison is clean
    for _, num, cat in airport_encoding.SETUPS.values():
        has_rates = set(airport_encoding.AIRPORT_RATES) <= set(num)
        has_categories = set(airport_encoding.AIRPORT_COLUMNS) <= set(cat)
        assert has_rates != has_categories


def test_airport_rates_ignore_the_test_year(db):
    before, _ = airport_encoding.airport_past_delay_rate(db, "Origin")
    # Make every 2008 flight very late: if 2008 leaked into the airport rates, they would change
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET ArrDelay = 999 WHERE Year = 2008")
    after, _ = airport_encoding.airport_past_delay_rate(db, "Origin")
    pd.testing.assert_frame_equal(before, after)


def test_small_airports_are_pulled_towards_the_overall_rate(db):
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO flightdetails (Year, Origin, Dest, ArrDelay) VALUES (2007, 'XXX', 'YYY', 60)")
    airports, overall = airport_encoding.airport_past_delay_rate(db, "Origin")
    tiny = airports.loc[airports["Origin"] == "XXX", "OriginPastDelayRate"].item()
    # One late flight alone would be a 100% rate; smoothing keeps it close to the average
    assert overall < tiny < overall + 0.05


def test_new_airports_get_the_overall_rate(db):
    flights = pd.DataFrame({"Origin": ["ORD", "NEW"], "Dest": ["LAX", "NEW"]})
    with_rates = airport_encoding.add_airport_rates(flights, db)
    _, overall = airport_encoding.airport_past_delay_rate(db, "Origin")
    assert with_rates.loc[1, "OriginPastDelayRate"] == overall
    assert with_rates[airport_encoding.AIRPORT_RATES].notna().all().all()


def test_cap_summary_counts_airports_beyond_the_limit():
    train = pd.DataFrame({"Origin": ["A", "A", "B", "C"], "Dest": ["X", "X", "X", "Y"]})
    cap = airport_encoding.airports_over_the_cap(train, max_categories=2)
    # One airport is kept, the rest share the "other" slot
    assert cap.loc["Origin", "airports_grouped_as_other"] == 2
    assert cap.loc["Origin", "share_of_training_flights_grouped"] == 0.5
    assert cap.loc["Dest", "share_of_training_flights_grouped"] == 0.25


def test_fit_and_score_runs_every_setup(db):
    # As in test_model_v4: make some planes land where they next depart, so the plane features exist
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET Dest = 'ORD' WHERE rowid % 3 = 0")
    df = airport_encoding.load(db, pct=100, seed=42)
    train, test = airport_encoding.model_v2.split_by_year(df)
    results, _ = airport_encoding.model_v3.fit_and_score(train, test, airport_encoding.SETUPS)
    assert list(results["model"])[1:] == list(airport_encoding.SETUPS)
    assert results["test_auc"].between(0, 1).all()
