"""
Flight delay prediction, v4: predict 2 hours before departure.

v2 and v3 only use the timetable, so they could make their prediction weeks
ahead. They top out at an AUC of about 0.65, because most delays come from
things that happen on the day. v4 moves the prediction to 2 hours before the
scheduled departure, and adds only what an airline would know by then:

- AirportLateShareEarlier: of the flights at the same airport scheduled to
  leave between 5 hours and 2h15m before ours, the share that left 15+ minutes
  late. A flight scheduled 2h15m before ours has either left by 2 hours before
  ours, or it is already known to be 15+ minutes late.
- PlaneTurnaroundMinutes: the scheduled gap between the same plane landing
  here on its previous flight and our departure. A short gap leaves no room
  to catch up.
- PlanePrevDepDelay: how many minutes late the same plane's previous flight
  took off, but only if it had taken off by 2 hours before our departure.
  Otherwise it is left blank, because we would not know it yet.

The leakage rule from v2 still holds: nothing that happens after the 2-hour
cutoff is used. tests/test_model_v4.py checks this by changing everything
after the cutoff and confirming the features do not move.

Same sample and split as v2 and v3: train on 2004-2007, test on 2008 (Jan-Apr).

Usage:
    python src/model_v4.py --db data/Fly.db [--pct 1] [--seed 42]
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_v2
import model_v3

ROOT = Path(__file__).resolve().parents[1]

CUTOFF_MINUTES = 120         # we predict 2 hours before scheduled departure
WINDOW_START_MINUTES = 300   # look at the airport's flights from 5 hours before ours...
LATE_MINUTES = 15            # ...up to the last one whose 15-minute lateness is known by the cutoff
# Placeholder tail numbers that many unrelated planes share in the data
JUNK_TAILS = {"0", "000000", "UNKNOW", "NA", ""}

AIRPORT_EARLIER = ["AirportLateShareEarlier"]
PLANE_PREVIOUS = ["PlaneTurnaroundMinutes", "PlanePrevDepDelay"]

V3_NUM = model_v3.V3_NUM
V4_NUM = V3_NUM + AIRPORT_EARLIER + PLANE_PREVIOUS
CAT = model_v2.PRE_DEPARTURE_CAT

SETUPS = {
    "v3 gradient boosting (timetable only)": (model_v2.boosting, V3_NUM, CAT),
    "v4 gradient boosting + airport's earlier delays only": (model_v2.boosting, V3_NUM + AIRPORT_EARLIER, CAT),
    "v4 gradient boosting + plane's previous flight only": (model_v2.boosting, V3_NUM + PLANE_PREVIOUS, CAT),
    "v4 logistic regression (2 hours before)": (model_v2.logistic, V4_NUM, CAT),
    "v4 gradient boosting (2 hours before)": (model_v2.boosting, V4_NUM, CAT),
}

# CROSS JOIN makes SQLite read flightdetails once and look each row up in the
# small table of sampled flights (as in model_v3).
AIRPORT_EARLIER_QUERY = f"""
SELECT s.FlightId, COUNT(*) AS EarlierFlights, SUM(f.DepDelay >= {LATE_MINUTES}) AS EarlierLate
FROM flightdetails f
CROSS JOIN sampled s
WHERE f.Year = s.Year AND f.Month = s.Month AND f.DayofMonth = s.DayofMonth AND f.Origin = s.Origin
  AND (f.CRSDepTime / 100) * 60 + f.CRSDepTime % 100
      BETWEEN s.DepMinutes - {WINDOW_START_MINUTES} AND s.DepMinutes - {CUTOFF_MINUTES + LATE_MINUTES}
  AND f.DepDelay IS NOT NULL
GROUP BY s.FlightId
"""

# Earlier flights by the same plane on the same day that landed at our departure airport
PLANE_EARLIER_QUERY = """
SELECT s.FlightId, f.CRSDepTime AS PrevCRSDepTime, f.CRSArrTime AS PrevCRSArrTime,
       f.ArrTime AS PrevArrTime, f.ActualElapsedTime AS PrevElapsed, f.DepDelay AS PrevDepDelay
FROM flightdetails f
CROSS JOIN sampled s
WHERE f.Year = s.Year AND f.Month = s.Month AND f.DayofMonth = s.DayofMonth
  AND f.TailNum = s.TailNum AND f.Dest = s.Origin AND f.CRSDepTime < s.CRSDepTime
"""


def to_minutes(hhmm):
    """Clock time like 1745 to minutes after midnight (1065)."""
    return (hhmm // 100) * 60 + hhmm % 100


def save_sampled_flights(conn, df):
    """Put the sampled flights in a temporary table so SQL can match other flights to them."""
    sampled = pd.DataFrame({
        "FlightId": df["FlightId"].astype(int).values,
        "Year": df["Year"].astype(int).values,
        "Month": df["Month"].astype(int).values,
        "DayofMonth": df["DayofMonth"].astype(int).values,
        "Origin": df["Origin"].values,
        "TailNum": df["TailNum"].where(~df["TailNum"].isin(JUNK_TAILS)).values,
        "CRSDepTime": df["CRSDepTime"].astype(int).values,
        "DepMinutes": to_minutes(df["CRSDepTime"].astype(int)).values,
    })
    conn.execute("DROP TABLE IF EXISTS temp.sampled")
    conn.execute("CREATE TEMP TABLE sampled (FlightId INTEGER PRIMARY KEY, Year INTEGER, Month INTEGER, "
                 "DayofMonth INTEGER, Origin TEXT, TailNum TEXT, CRSDepTime INTEGER, DepMinutes INTEGER)")
    conn.executemany("INSERT INTO sampled VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     sampled.astype(object).where(sampled.notna(), None).itertuples(index=False))
    # One index per way we look flights up: by airport and day, and by plane and day
    conn.execute("CREATE INDEX temp.sampled_airport ON sampled (Year, Month, DayofMonth, Origin)")
    conn.execute("CREATE INDEX temp.sampled_plane ON sampled (Year, Month, DayofMonth, TailNum)")


def airport_earlier_delays(conn):
    """Share of the airport's earlier flights already known to be 15+ min late at the cutoff."""
    earlier = pd.read_sql_query(AIRPORT_EARLIER_QUERY, conn)
    earlier["AirportLateShareEarlier"] = earlier["EarlierLate"] / earlier["EarlierFlights"]
    return earlier[["FlightId", "AirportLateShareEarlier"]]


def plane_previous_flight(conn):
    """The plane's previous flight into our airport: turnaround gap, and its delay if already known."""
    candidates = pd.read_sql_query(PLANE_EARLIER_QUERY, conn)
    # The previous flight is the latest of the plane's earlier flights that day
    previous = (candidates.sort_values(["FlightId", "PrevCRSDepTime"])
                .drop_duplicates("FlightId", keep="last"))
    return previous


def add_same_day_features(df, db_path):
    """Add the three features an airline would know 2 hours before departure."""
    with sqlite3.connect(db_path) as conn:
        save_sampled_flights(conn, df)
        earlier = airport_earlier_delays(conn)
        previous = plane_previous_flight(conn)

    df = df.merge(earlier, on="FlightId", how="left")
    df = df.merge(previous, on="FlightId", how="left")
    dep_minutes = to_minutes(df["CRSDepTime"])

    # Both times are in our airport's local clock, so the gap needs no time-zone fix
    turnaround = dep_minutes - to_minutes(df["PrevCRSArrTime"])
    df["PlaneTurnaroundMinutes"] = turnaround.where(turnaround >= 0)

    # When did the previous flight take off, in our airport's clock? Landing time minus
    # flight time. A landing time "earlier" than its departure means it landed after midnight.
    landed = to_minutes(df["PrevArrTime"])
    landed = landed.where(landed >= to_minutes(df["PrevCRSDepTime"]), landed + 24 * 60)
    took_off = landed - df["PrevElapsed"]
    known_by_cutoff = took_off <= dep_minutes - CUTOFF_MINUTES
    df["PlanePrevDepDelay"] = df["PrevDepDelay"].where(known_by_cutoff)
    return df.drop(columns=["PrevCRSDepTime", "PrevCRSArrTime", "PrevArrTime", "PrevElapsed", "PrevDepDelay"])


def coverage(test):
    """How often each same-day feature is available for a test flight."""
    return {col: test[col].notna().mean() for col in AIRPORT_EARLIER + PLANE_PREVIOUS}


def plot_riskiest(results, ax=None):
    """Bar chart: share of flights late, on average vs in each model's riskiest 10%."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    v3_name, v4_name = list(SETUPS)[0], list(SETUPS)[-1]
    rows = results.set_index("model")
    bars = pd.Series({
        "All 2008 flights": rows.loc[v4_name, "late_rate_overall"],
        "Riskiest 10%: timetable\nonly (v3)": rows.loc[v3_name, "late_rate_in_flagged"],
        "Riskiest 10%: 2 hours\nbefore (v4)": rows.loc[v4_name, "late_rate_in_flagged"],
    })
    colors = ["#9aa0a6", "#7ba7d9", "#1f5fa8"]
    ax.barh(bars.index[::-1], bars.values[::-1] * 100, color=colors[::-1])
    for y, value in enumerate(bars.values[::-1]):
        ax.text(value * 100 + 0.5, y, f"{value:.1%}", va="center")
    ax.set_xlabel("Share of flights arriving 15+ minutes late (%)")
    ax.set_title("Flagging the riskiest 10% of flights (test set: Jan-Apr 2008)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, bars.max() * 100 * 1.2)
    ax.figure.tight_layout()
    return ax


def load(db_path, pct, seed):
    """v3's sample and features plus the same-day features."""
    df = model_v3.load(db_path, pct, seed)
    return add_same_day_features(df, db_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = load(args.db, args.pct, args.seed)
    train, test = model_v2.split_by_year(df)
    print(f"Train {len(train):,} rows | Test {len(test):,} rows")
    for col, share in coverage(test).items():
        print(f"{col}: known for {share:.1%} of test flights")

    results, _ = model_v3.fit_and_score(train, test, SETUPS)
    results["pct"], results["seed"] = args.pct, args.seed
    pd.set_option("display.width", 200)
    print(results[["model", "test_auc", "late_rate_in_flagged", "times_the_average",
                   "share_of_late_flights_caught"]].round(3).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.round(4).to_csv(out / "model_comparison_v4.csv", index=False)
    pd.Series(coverage(test), name="share_of_test_flights").round(4).to_csv(out / "feature_coverage_v4.csv")
    ax = plot_riskiest(results)
    ax.figure.savefig(ROOT / "images" / "riskiest_flights_v4.png", dpi=150)
    print(f"\nSaved results/model_comparison_v4.csv, results/feature_coverage_v4.csv "
          f"and images/riskiest_flights_v4.png")


if __name__ == "__main__":
    main()
