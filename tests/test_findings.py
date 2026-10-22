"""Tests for src/findings.py on a tiny hand-made database (no real data needed)."""

import sqlite3

import pandas as pd
import pytest

import findings


@pytest.fixture
def db(tmp_path):
    """Eight flights covering every day, season and age group, with known delays."""
    flights = pd.DataFrame({
        "Year":      [2006, 2006, 2006, 2006, 2006, 2006, 2006, 2006],
        "Month":     [1, 4, 7, 10, 12, 2, 6, 3],
        "DayOfWeek": [1, 2, 3, 4, 5, 6, 7, 1],
        "ArrDelay":  [30.0, 0.0, 20.0, 0.0, 60.0, 0.0, 0.0, None],  # last flight never arrived
        "TailNum":   ["NEW", "MID", "OLD", "ZERO", "NOYEAR", "GHOST", "NEW", "NEW"],
    })
    planes = pd.DataFrame({
        # ZERO has build year 0, the bug the notebook had; NOYEAR has no year at all.
        "tailnum": ["NEW", "MID", "OLD", "ZERO", "NOYEAR"],
        "year":    [2000.0, 1990.0, 1980.0, 0.0, None],
    })
    path = tmp_path / "Fly.db"
    with sqlite3.connect(path) as conn:
        flights.to_sql("flightdetails", conn, index=False)
        planes.to_sql("planes", conn, index=False)
    return path


def test_build_year_zero_is_unknown_not_old(db):
    ages = findings.all_findings(db)["aircraft_age"].set_index("Group")
    assert ages.loc["Over 20 years", "Flights"] == 1
    # ZERO (year 0), NOYEAR (no year) and GHOST (not in planes table) are all unknown.
    assert ages.loc["Unknown", "Flights"] == 3


def test_age_groups(db):
    ages = findings.all_findings(db)["aircraft_age"].set_index("Group")
    assert ages.loc["0-10 years", "Rate"] == 0.5
    assert ages.loc["11-20 years", "Rate"] == 0.0
    assert ages.loc["Over 20 years", "Rate"] == 1.0


def test_december_counts_as_winter(db):
    seasons = findings.all_findings(db)["season"].set_index("Group")
    assert seasons.loc["Winter", "Flights"] == 3  # January, December, February
    assert seasons.loc["Winter", "Rate"] == pytest.approx(2 / 3)


def test_days_in_week_order_and_unarrived_flight_left_out(db):
    days = findings.all_findings(db)["day_of_week"]
    assert list(days["Group"]) == findings.DAYS
    assert days["Flights"].sum() == 7


def test_charts_and_tables_are_saved(db, tmp_path):
    result = findings.all_findings(db)
    findings.save_tables(result, tmp_path / "results")
    findings.draw_all_charts(result, tmp_path)
    assert (tmp_path / "results" / "findings_season.csv").exists()
    assert (tmp_path / "delay_rate_by_aircraft_age.png").stat().st_size > 0
