"""Tests for src/explain_flights.py on the random database in conftest.py."""

import sqlite3

import numpy as np
import pandas as pd

import explain_flights
import model_v3


def test_groups_cover_every_model_column_once():
    explain_flights.check_groups()


class FakeModel:
    """Chance of delay = previous flight's delay / 100, so only the 'plane' group matters."""

    def predict_proba(self, X):
        p = X["PlanePrevDepDelay"].fillna(0).clip(0, 100).to_numpy() / 100
        return np.c_[1 - p, p]


def test_only_the_group_the_model_uses_gets_a_push():
    flights = pd.DataFrame({col: [1.0, 2.0] for col in explain_flights.FEATURES})
    flights["PlanePrevDepDelay"] = [60.0, 0.0]
    references = flights.assign(PlanePrevDepDelay=[10.0, 10.0])
    pushes = explain_flights.group_pushes(FakeModel(), flights, references)
    # 60 minutes late against an ordinary 10: the chance is 50 points higher; 0 is 10 lower
    assert np.allclose(pushes["plane"], [50, -10])
    assert np.allclose(pushes.drop(columns="plane"), 0)
    reasons = explain_flights.top_reasons(pushes)
    assert reasons == [[["plane", 50]], [["plane", -10]]]


def test_reasons_are_added_to_the_matching_page_rows():
    page = {"columns": ["date", "carrier", "flight_number", "origin", "dest", "scheduled_departure"],
            "flights": [["2008-01-03", "AA", 12, "ORD", "LAX", 905], ["2008-01-04", "UA", 7, "SFO", "JFK", 1300]]}
    flights = pd.DataFrame({"Year": [2008], "Month": [1], "DayofMonth": [3], "UniqueCarrier": ["AA"],
                            "FlightNum": [12], "Origin": ["ORD"], "Dest": ["LAX"], "CRSDepTime": [905]})
    matched = explain_flights.add_to_page(page, explain_flights.page_key(flights), [[["plane", 20]]])
    assert matched == 1
    assert page["columns"][-1] == "reasons_2h"
    assert page["flights"][0][-1] == [["plane", 20]] and page["flights"][1][-1] is None


def test_end_to_end_on_the_small_database(db):
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET Dest = 'ORD' WHERE rowid % 3 = 0")
    model, train, test = explain_flights.train_2h_model(model_v3.load(db, pct=100, seed=42), db)
    pushes = explain_flights.group_pushes(model, test.head(30), train.sample(5, random_state=0))
    assert list(pushes.columns) == list(explain_flights.GROUPS) and len(pushes) == 30
    summary = explain_flights.reason_summary(pushes)
    assert np.isclose(summary["share_biggest_reason"].sum(), 1)
