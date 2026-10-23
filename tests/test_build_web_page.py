"""Tests for src/build_web_page.py on the random database in conftest.py."""

import json
import sqlite3

import pandas as pd

import build_web_page
import model_v2
import model_v3


def page_data(db):
    # The random database never lands a plane where it next departs from; make some do
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET Dest = 'ORD' WHERE rowid % 3 = 0")
        pd.DataFrame({"Code": ["AA"], "Description": ["American Airlines"]}).to_sql("carriers", conn, index=False)
        pd.DataFrame({"iata": ["ORD"], "city": ["Chicago"], "state": ["IL"]}).to_sql("airports", conn, index=False)
    df = model_v3.load(db, pct=100, seed=42)
    _, test = model_v2.split_by_year(df)
    risks, facts = build_web_page.predictions_at_each_time(df, db)
    return test, risks, build_web_page.page_rows(test, risks, facts), build_web_page.names(db)


def test_every_test_flight_has_a_row_with_every_column(db):
    test, _, rows, _ = page_data(db)
    assert len(rows) == len(test)
    assert all(len(row) == len(build_web_page.COLUMNS) for row in rows)


def test_only_2008_flights_are_shown(db):
    _, _, rows, _ = page_data(db)
    assert {row[0][:4] for row in rows} == {"2008"}


def test_rows_can_be_saved_as_json(db):
    # Unknown values must be null, not NaN, or the browser cannot read the file
    _, _, rows, _ = page_data(db)
    assert "NaN" not in json.dumps(rows)


def test_one_risk_per_prediction_time(db):
    _, risks, _, _ = page_data(db)
    assert list(risks.columns) == list(build_web_page.PREDICTION_TIMES)
    assert risks.stack().between(0, 1).all()


def test_airline_and_airport_names_are_looked_up(db):
    _, _, _, (carriers, airports) = page_data(db)
    assert carriers["AA"] == "American Airlines"
    assert airports["ORD"] == "Chicago, IL"
