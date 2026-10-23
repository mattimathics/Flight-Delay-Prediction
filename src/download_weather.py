"""
Download hourly airport weather reports for the busiest airports, 2004 to April 2008.

Source: the Iowa Environmental Mesonet (Iowa State University), a free public
archive of the official weather reports every US airport issues at least once
an hour. https://mesonet.agron.iastate.edu/request/download.phtml

Only a few columns are kept: visibility (miles), wind and gust speed (knots),
rain in the last hour (inches), the height of the lowest cloud layer (feet)
and the weather codes (TS = thunderstorm, SN = snow, FG = fog, ...).
Times are saved in each airport's local time, to match the flight timetable.

One CSV per airport is saved to the output folder. Airports that are already
downloaded are skipped, so the script can be stopped and rerun.

Usage:
    python src/download_weather.py --db data/Fly.db [--airports 100] [--out data/weather]
"""

import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = "https://mesonet.agron.iastate.edu"
COLUMNS = ["vsby", "sknt", "gust", "p01i", "skyl1", "wxcodes"]
FIRST_DAY, LAST_DAY = (2004, 1, 1), (2008, 5, 1)  # the flight data ends in April 2008
PAUSE_SECONDS = 5  # be polite to a free service: one request at a time, with a short pause


def busiest_airports(db_path, how_many):
    """Departure airports with the most flights, and the state each is in."""
    with sqlite3.connect(db_path) as conn:
        return conn.execute("""
            SELECT f.Origin, a.state
            FROM flightdetails f LEFT JOIN airports a ON a.iata = f.Origin
            GROUP BY f.Origin ORDER BY COUNT(*) DESC LIMIT ?""", (how_many,)).fetchall()


def fetch(url):
    """Download a URL with curl, which retries on its own if the connection drops.

    Python's own downloader kept being cut off by this server on larger files,
    while curl (built into Windows 10+, macOS and Linux) downloads them fine.
    """
    result = subprocess.run(["curl", "--silent", "--show-error", "--fail", "--max-time", "600",
                             "--retry", "5", "--retry-all-errors", "--retry-delay", "10", url],
                            capture_output=True, check=True)
    return result.stdout


def time_zones(states):
    """Each weather station's time zone, from the archive's station list for each state."""
    zones = {}
    for state in sorted(set(states)):
        try:
            stations = json.loads(fetch(f"{ARCHIVE}/geojson/network/{state}_ASOS.geojson"))
        except subprocess.CalledProcessError as problem:
            print(f"No station list for {state} ({problem}); its airports will be skipped")
            continue
        print(f"Station list for {state}: {len(stations['features'])} stations")
        for station in stations["features"]:
            zones[station["id"]] = station["properties"]["tzname"]
        time.sleep(PAUSE_SECONDS)
    return zones


def station_for(code, zones):
    """The weather station ID for an airport code, or None if there isn't one.

    Mainland airports use the same 3-letter code (ORD). Alaska and Hawaii use a
    4-letter code starting with P (HNL -> PHNL), and Puerto Rico starts with TJ.
    """
    for station in (code, "P" + code, "TJ" + code[:2]):
        if station in zones:
            return station
    return None


def weather_url(station, zone):
    """Download link for one station's reports, in local time."""
    params = [f"station={station}"] + [f"data={c}" for c in COLUMNS] + [
        f"year1={FIRST_DAY[0]}", f"month1={FIRST_DAY[1]}", f"day1={FIRST_DAY[2]}",
        f"year2={LAST_DAY[0]}", f"month2={LAST_DAY[1]}", f"day2={LAST_DAY[2]}",
        f"tz={zone}", "format=onlycomma", "latlon=no", "missing=null", "trace=0.0001",
        # Routine hourly reports, plus the extra ones issued when weather changes suddenly
        "report_type=3", "report_type=4",
    ]
    return f"{ARCHIVE}/cgi-bin/request/asos.py?" + "&".join(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--airports", type=int, default=100, help="how many of the busiest airports (default 100)")
    ap.add_argument("--out", default=str(ROOT / "data" / "weather"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    airports = busiest_airports(args.db, args.airports)
    # The station time zones are saved once, so a rerun does not fetch them again
    zones_file = out / "time_zones.json"
    if zones_file.exists():
        zones = json.loads(zones_file.read_text())
    else:
        zones = time_zones(state for _, state in airports if state)
        zones_file.write_text(json.dumps(zones))

    for number, (code, _) in enumerate(airports, start=1):
        target = out / f"{code}.csv"
        if target.exists():
            continue
        station = station_for(code, zones)
        if station is None:
            print(f"{number:3}. {code}: no weather station found, skipped")
            continue
        try:
            content = fetch(weather_url(station, zones[station]))
        except subprocess.CalledProcessError as problem:
            # One failed airport should not stop the others; rerunning fills the gap
            print(f"{number:3}. {code}: download failed ({problem.stderr.decode().strip()}), skipped")
            continue
        target.write_bytes(content)
        print(f"{number:3}. {code}: {content.count(b'\n') - 1:,} reports, {len(content) / 1e6:.1f} MB")
        time.sleep(PAUSE_SECONDS)

    total = sum(f.stat().st_size for f in out.glob("*.csv"))
    print(f"\n{len(list(out.glob('*.csv')))} airports saved in {out} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
