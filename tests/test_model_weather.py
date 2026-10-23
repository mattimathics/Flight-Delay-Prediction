"""Tests for src/model_weather.py with a few hand-made weather reports."""

import json

import numpy as np
import pandas as pd
import pytest

import model_weather
from test_model_v2 import POST_DEPARTURE

HEADER = "station,valid,vsby,sknt,gust,p01i,skyl1,wxcodes\n"


@pytest.fixture
def folder(tmp_path):
    # ORD: a clear report at 15:50, then a thunderstorm at 16:10 (after the 16:00 cutoff)
    (tmp_path / "ORD.csv").write_text(HEADER
        + "ORD,2007-01-10 15:50,10.00,8.00,null,0.0000,25000.00,null\n"
        + "ORD,2007-01-10 16:10,1.00,30.00,45.00,0.5000,800.00,+TSRA\n")
    # LAX (2 hours behind Chicago): fog at 10:00, a storm at 15:00, both Los Angeles time
    (tmp_path / "LAX.csv").write_text(HEADER
        + "LAX,2007-01-10 10:00,5.00,4.00,null,null,3000.00,BR FG\n"
        + "LAX,2007-01-10 15:00,2.00,25.00,40.00,0.3000,900.00,TSRA\n")
    (tmp_path / "time_zones.json").write_text(json.dumps({
        "ORD": "America/Chicago", "LAX": "America/Los_Angeles", "JFK": "America/New_York"}))
    return tmp_path


def flight(origin="ORD", dest="LAX", dep=1800):
    return pd.DataFrame({"Year": [2007], "Month": ["1"], "DayofMonth": [10], "CRSDepTime": [dep],
                         "Origin": [origin], "Dest": [dest]})


def test_no_post_departure_feature_in_any_setup():
    for _, num, cat in model_weather.SETUPS.values():
        assert not set(num + cat) & POST_DEPARTURE


def test_uses_the_last_report_before_the_cutoff(folder):
    row = model_weather.add_weather(flight(), folder).iloc[0]
    assert row["OriginVisibility"] == 10
    assert row["OriginWind"] == 8
    assert row["OriginStorm"] == 0  # the 16:10 storm came after the cutoff


def test_a_later_flight_sees_the_storm(folder):
    row = model_weather.add_weather(flight(dep=1830), folder).iloc[0]  # cutoff 16:30
    assert row["OriginStorm"] == 1
    assert row["OriginGust"] == 45


def test_stale_reports_are_ignored(folder):
    # Cutoff 16:00 Chicago = 14:00 Los Angeles: the 10:00 fog report is 4 hours old
    row = model_weather.add_weather(flight(), folder).iloc[0]
    assert np.isnan(row["DestVisibility"])


def test_destination_weather_uses_the_destination_clock(folder):
    # Leaves New York at 16:30, so the cutoff is 14:30 New York = 11:30 Los Angeles.
    # The 15:00 Los Angeles storm is hours after the cutoff and must not be used.
    row = model_weather.add_weather(flight(origin="JFK", dep=1630), folder).iloc[0]
    assert row["DestFog"] == 1
    assert row["DestStorm"] == 0


def test_airports_without_weather_get_blanks(folder):
    row = model_weather.add_weather(flight(origin="XXX"), folder).iloc[0]
    assert np.isnan(row["OriginVisibility"])


def test_weather_codes_become_flags(folder):
    zones = model_weather.airport_time_zones(folder, ["ORD", "LAX"])
    reports = model_weather.read_reports(folder, zones)
    lax = reports[reports["Airport"] == "LAX"].iloc[0]
    assert (lax["Fog"], lax["Storm"], lax["Snow"]) == (1, 0, 0)


def test_flight_order_is_kept(folder):
    flights = pd.concat([flight(dep=1830), flight(dep=1800)], ignore_index=True)
    result = model_weather.add_weather(flights, folder)
    assert list(result["OriginStorm"]) == [1, 0]


def test_station_codes_outside_the_mainland():
    import download_weather
    zones = {"ORD": "America/Chicago", "PHNL": "Pacific/Honolulu", "TJSJ": "America/Puerto_Rico"}
    assert download_weather.station_for("ORD", zones) == "ORD"
    assert download_weather.station_for("HNL", zones) == "PHNL"
    assert download_weather.station_for("SJU", zones) == "TJSJ"
    assert download_weather.station_for("XXX", zones) is None
