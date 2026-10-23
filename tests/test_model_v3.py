"""Tests for src/model_v3.py on the tiny synthetic database in conftest.py."""

import sqlite3

import pandas as pd

import model_v3
from test_model_v2 import POST_DEPARTURE


def test_no_post_departure_feature_in_any_setup():
    for _, num, cat in model_v3.SETUPS.values():
        assert not set(num + cat) & POST_DEPARTURE


def test_busyness_matches_a_direct_count(db):
    df = model_v3.load(db, pct=100, seed=42)
    with sqlite3.connect(db) as conn:
        everything = pd.read_sql_query("SELECT * FROM flightdetails", conn)
    flight = df.iloc[0]
    same_slot = everything[(everything["Year"] == flight["Year"])
                           & (everything["Month"] == int(flight["Month"]))
                           & (everything["DayofMonth"] == flight["DayofMonth"])
                           & (everything["Origin"] == flight["Origin"])
                           & ((everything["CRSDepTime"] // 100).clip(upper=23) == flight["CRSDepHour"])]
    assert flight["OriginHourFlights"] == len(same_slot)


def test_busyness_counts_cancelled_flights(db):
    # Cancelled flights were still on the timetable, so they count towards how busy a slot is
    flight = model_v3.load(db, pct=100, seed=42).iloc[0]
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO flightdetails (Year, Month, DayofMonth, Origin, CRSDepTime, ArrDelay) "
                     "VALUES (?, ?, ?, ?, ?, NULL)",
                     (int(flight["Year"]), int(flight["Month"]), int(flight["DayofMonth"]),
                      flight["Origin"], int(flight["CRSDepTime"])))
    flight_after = model_v3.load(db, pct=100, seed=42).iloc[0]
    assert flight_after["OriginHourFlights"] == flight["OriginHourFlights"] + 1


def test_route_history_ignores_the_test_year(db):
    before, _ = model_v3.route_past_delay_rate(db)
    # Make every 2008 flight very late: if 2008 leaked into the route rates, they would change
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET ArrDelay = 999 WHERE Year = 2008")
    after, _ = model_v3.route_past_delay_rate(db)
    pd.testing.assert_frame_equal(before, after)


def test_small_routes_are_pulled_towards_the_overall_rate(db):
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO flightdetails (Year, Origin, Dest, ArrDelay) VALUES (2007, 'XXX', 'YYY', 60)")
    routes, overall = model_v3.route_past_delay_rate(db)
    tiny = routes.loc[(routes["Origin"] == "XXX") & (routes["Dest"] == "YYY"), "RoutePastDelayRate"].item()
    # One late flight alone would be a 100% rate; smoothing keeps it close to the average
    assert overall < tiny < overall + 0.05


def test_flag_riskiest_on_a_known_example():
    late = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    risk = [0.9, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    result = model_v3.flag_riskiest(late, risk, share=0.1)
    assert result["flagged_flights"] == 1
    assert result["late_rate_in_flagged"] == 1.0
    assert result["share_of_late_flights_caught"] == 1 / 3
    assert result["times_the_average"] == 1.0 / 0.3


def test_fit_and_score_has_benchmark_and_every_setup(db):
    import matplotlib
    matplotlib.use("Agg")
    train, test = model_v3.model_v2.split_by_year(model_v3.load(db, pct=100, seed=42))
    results, _ = model_v3.fit_and_score(train, test)
    assert list(results["model"]) == [model_v3.BENCHMARK] + list(model_v3.SETUPS)
    assert results["test_auc"].between(0, 1).all()
    assert results["late_rate_in_flagged"].between(0, 1).all()
    ax = model_v3.plot_riskiest(results)
    assert len(ax.patches) == 4
