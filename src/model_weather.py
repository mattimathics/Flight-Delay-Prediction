"""
Does weather help the 2-hours-before model?

For each flight, this adds the last weather report issued at the departure
and arrival airports *before* the 2-hour cutoff: visibility, wind, gusts,
rain in the last hour, the lowest cloud height, and whether a thunderstorm,
snow or fog was reported. Weather at the actual departure time is never
used, because it is not known 2 hours ahead.

The reports come from src/download_weather.py (the 100 busiest airports,
which cover 92% of departures). Flights at other airports get blanks.

Same sample and split as v2-v4: train on 2004-2007, test on 2008 (Jan-Apr).

Usage:
    python src/model_weather.py --db data/Fly.db --weather data/weather [--pct 1] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import model_v2
import model_v3
import model_v4
from download_weather import station_for

ROOT = Path(__file__).resolve().parents[1]

# A report older than this before the cutoff is too stale to describe the weather then
MAX_REPORT_AGE = pd.Timedelta(hours=3)
REPORT_COLUMNS = ["Visibility", "Wind", "Gust", "Rain", "CloudBase", "Storm", "Snow", "Fog"]

ORIGIN_WEATHER = [f"Origin{c}" for c in REPORT_COLUMNS]
DEST_WEATHER = [f"Dest{c}" for c in REPORT_COLUMNS]
WEATHER = ORIGIN_WEATHER + DEST_WEATHER

CAT = model_v2.PRE_DEPARTURE_CAT
SETUPS = {
    "Timetable only (v3)": (model_v2.boosting, model_v4.V3_NUM, CAT),
    # Weather is only known on the day, so this is a 2-hour prediction, not a weeks-ahead one
    "Timetable + weather (2 hours before)": (model_v2.boosting, model_v4.V3_NUM + WEATHER, CAT),
    "2 hours before (v4)": (model_v2.boosting, model_v4.V4_NUM, CAT),
    "2 hours before + weather": (model_v2.boosting, model_v4.V4_NUM + WEATHER, CAT),
}


def airport_time_zones(folder, airports):
    """Each airport's time zone, from the station list saved by download_weather.py."""
    zones = json.loads((Path(folder) / "time_zones.json").read_text())
    found = {code: station_for(code, zones) for code in set(airports)}
    return {code: zones[station] for code, station in found.items() if station}


def to_utc(local_times, zone):
    """Local clock times at one airport to world time (UTC), so airports can be compared."""
    # Clock times that happen twice or never when clocks change for summer time become blank
    utc = local_times.dt.tz_localize(zone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return utc.astype("datetime64[ns, UTC]")


def read_reports(folder, zones):
    """All downloaded weather reports, one row per report, with plain column names."""
    frames = []
    for path in sorted(Path(folder).glob("*.csv")):
        if path.stem not in zones:
            continue
        raw = pd.read_csv(path, na_values=["null"], low_memory=False)
        codes = raw["wxcodes"].fillna("").astype(str)
        frames.append(pd.DataFrame({
            "Airport": path.stem,
            "Time": to_utc(pd.to_datetime(raw["valid"]), zones[path.stem]),
            "Visibility": raw["vsby"],
            "Wind": raw["sknt"],
            "Gust": raw["gust"],
            "Rain": raw["p01i"],
            "CloudBase": raw["skyl1"],
            # Weather codes are short tags such as "-RA BR" (light rain, mist) or "TSRA" (thunderstorm, rain)
            "Storm": codes.str.contains("TS").astype(int),
            "Snow": codes.str.contains("SN").astype(int),
            "Fog": codes.str.contains("FG").astype(int),
        }))
    return pd.concat(frames, ignore_index=True).dropna(subset=["Time"]).sort_values("Time")


def cutoff_times(df, zones, cutoff_minutes=model_v4.CUTOFF_MINUTES):
    """The moment (in UTC) 2 hours before each flight's scheduled departure.

    The timetable is in the departure airport's local time. Converting to UTC
    matters for the destination's weather: 16:00 in New York is only 13:00 in
    Los Angeles, so a Los Angeles report from 15:00 local time is in the future.
    """
    dates = pd.to_datetime(dict(year=df["Year"].astype(int), month=df["Month"].astype(int),
                                day=df["DayofMonth"].astype(int)))
    minutes = model_v4.to_minutes(df["CRSDepTime"].astype(int))
    local = dates + pd.to_timedelta(minutes - cutoff_minutes, unit="min")
    utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for airport, zone in zones.items():
        at_airport = df["Origin"] == airport
        if at_airport.any():
            utc[at_airport] = to_utc(local[at_airport], zone)
    return utc


def last_report_before(flights, reports, airport_col, prefix):
    """For each flight, the latest report at that airport issued at or before the cutoff."""
    left = pd.DataFrame({"Row": range(len(flights)), "Airport": flights[airport_col].values,
                         "Time": flights["Cutoff"].reset_index(drop=True)})
    # Flights whose cutoff time is unknown cannot be matched to any report
    left = left.dropna(subset=["Time"]).sort_values("Time")
    matched = pd.merge_asof(left, reports, on="Time", by="Airport",
                            direction="backward", tolerance=MAX_REPORT_AGE)
    matched = matched.set_index("Row").reindex(range(len(flights)))
    return matched[REPORT_COLUMNS].rename(columns=lambda c: prefix + c).set_index(flights.index)


def add_weather(df, folder):
    """Add the origin's and destination's last weather report before the cutoff."""
    zones = airport_time_zones(folder, pd.concat([df["Origin"], df["Dest"]]))
    reports = read_reports(folder, zones)
    df = df.assign(Cutoff=cutoff_times(df, zones))
    df = df.join(last_report_before(df, reports, "Origin", "Origin"))
    df = df.join(last_report_before(df, reports, "Dest", "Dest"))
    return df.drop(columns="Cutoff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--weather", default=str(ROOT / "data" / "weather"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = add_weather(model_v4.load(args.db, args.pct, args.seed), args.weather)
    train, test = model_v2.split_by_year(df)
    print(f"Train {len(train):,} rows | Test {len(test):,} rows")
    print(f"Origin weather known for {test['OriginVisibility'].notna().mean():.1%} of test flights, "
          f"destination weather for {test['DestVisibility'].notna().mean():.1%}")

    results, _ = model_v3.fit_and_score(train, test, SETUPS)
    results["pct"], results["seed"] = args.pct, args.seed
    pd.set_option("display.width", 200)
    print(results[["model", "test_auc", "late_rate_in_flagged", "times_the_average",
                   "share_of_late_flights_caught"]].round(3).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.round(4).to_csv(out / "model_comparison_weather.csv", index=False)
    print("\nSaved results/model_comparison_weather.csv")


if __name__ == "__main__":
    main()
