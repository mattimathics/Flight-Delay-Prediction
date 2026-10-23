"""Tests for src/model_v2.py on a tiny synthetic database (no real data needed)."""

import pandas as pd
import pytest

import model_v2

POST_DEPARTURE = {"DepTime", "DepDelay", "TaxiOut", "TaxiIn", "AirTime", "ArrTime", "ArrDelay",
                  "ActualElapsedTime", "CarrierDelay", "WeatherDelay", "NASDelay",
                  "SecurityDelay", "LateAircraftDelay", "Cancelled", "Diverted"}


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


def test_fit_and_score_returns_one_auc_per_setup(db):
    import matplotlib
    matplotlib.use("Agg")
    train, test = model_v2.split_by_year(model_v2.load(db, pct=100))
    assert train["Year"].max() <= 2007
    assert set(test["Year"]) == {2008}
    results, fitted = model_v2.fit_and_score(train, test)
    assert list(results["model"]) == list(model_v2.SETUPS)
    assert results["test_auc"].between(0, 1).all()
    assert (results["n_test"] == len(test)).all()
    ax = model_v2.plot_roc(fitted, test)
    assert len(ax.get_lines()) == len(model_v2.SETUPS) + 1  # one curve per model + diagonal
