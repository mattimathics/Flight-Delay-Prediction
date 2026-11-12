"""
Does the warning list still work if it is made fresh each day?

The headline "74 in 100 flights on the warning list were late" flags the riskiest 10% of
all Jan-Apr 2008 test flights at once. An airline can't do that: on 3 January it doesn't
know 4 April's flights. It would flag the riskiest 10% of each day's flights instead.

That can change the result. On a stormy day many flights are risky, but a per-day list
still flags only 1 in 10 of them; on a calm day it must flag 1 in 10 even if none look
risky. This script scores both versions on the same saved predictions, so the only
difference is how the list is made.

It reads the predictions saved by src/error_bars.py (no retraining) and looks up each
flight's date in the database.

Usage:
    python src/per_day_warning_list.py --db data/Fly.db
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "data" / "test_predictions.csv.gz"
SHARE = 0.10  # the warning list: the riskiest 1 in 10 flights, as everywhere else in the project

# The models to score (columns of the saved predictions), in the README's order
MODELS = ["benchmark", "Weeks ahead", "4 hours before", "2 hours before", "1 hour before"]


def flight_dates(db_path, flight_ids):
    """The date of each test flight, looked up by its row id in the flights table."""
    with sqlite3.connect(db_path) as con:
        dates = pd.read_sql_query(
            "SELECT rowid AS FlightId, Year, Month, DayofMonth FROM flightdetails WHERE Year = 2008", con)
    dates["day"] = pd.to_datetime(dates[["Year", "Month", "DayofMonth"]].rename(
        columns={"Year": "year", "Month": "month", "DayofMonth": "day"}))
    return dates.set_index("FlightId").loc[flight_ids, "day"]


def whole_period_flags(risk, share=SHARE):
    """Flag the riskiest share of all flights at once (the published warning list)."""
    risk = pd.Series(risk).reset_index(drop=True)
    n = int(round(len(risk) * share))
    # Ties are broken by row order, as in model_v3.flag_riskiest, so the README's numbers are reproduced
    flagged = risk.sort_values(ascending=False, kind="stable").index[:n]
    mask = np.zeros(len(risk), dtype=bool)
    mask[flagged] = True
    return mask


def per_day_flags(risk, day, share=SHARE):
    """Flag the riskiest share of each day's flights separately, as an airline would each morning."""
    frame = pd.DataFrame({"risk": np.asarray(risk), "day": np.asarray(day)})
    # Rank within each day, riskiest first; "first" breaks ties by row order, as above
    rank = frame.groupby("day")["risk"].rank(ascending=False, method="first")
    per_day = frame.groupby("day")["risk"].transform("size")
    return (rank <= (per_day * share).round()).to_numpy()


def score(late, flagged, day):
    """How well one warning list works overall, and how much it varies from day to day."""
    late = np.asarray(late).astype(bool)
    daily = pd.DataFrame({"late": late, "flagged": flagged, "day": np.asarray(day)})
    on_list = daily[daily["flagged"]]
    daily_rate = on_list.groupby("day")["late"].mean()
    return {
        "flagged_flights": int(flagged.sum()),
        "late_rate_on_list": late[flagged].mean(),
        "share_of_late_flights_caught": late[flagged].sum() / late.sum(),
        "days_with_flights_on_list": int(daily_rate.size),
        # Middle 80% of days: how far a typical day's list strays from the overall rate
        "daily_late_rate_p10": daily_rate.quantile(0.10),
        "daily_late_rate_median": daily_rate.median(),
        "daily_late_rate_p90": daily_rate.quantile(0.90),
    }


def compare(table, day, models=MODELS, share=SHARE):
    """One row per model and list type: the whole-period list against a fresh list each day."""
    rows = []
    for model in [m for m in models if m in table]:
        risk = table[model].to_numpy()
        for how, flags in [("Whole Jan-Apr at once (published)", whole_period_flags(risk, share)),
                           ("Fresh list each day", per_day_flags(risk, day, share))]:
            rows.append({"model": model, "warning_list": how, **score(table["late"], flags, day)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--predictions", default=str(SAVED), help="saved by src/error_bars.py")
    args = ap.parse_args()

    table = pd.read_csv(args.predictions, index_col="FlightId")
    day = flight_dates(args.db, table.index).to_numpy()
    result = compare(table, day)

    pd.set_option("display.width", 200)
    print(f"{len(table):,} test flights over {pd.Series(day).nunique()} days")
    print(result.round(4).to_string(index=False))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    result.round(4).to_csv(out / "per_day_warning_list.csv", index=False)
    print("\nSaved results/per_day_warning_list.csv")


if __name__ == "__main__":
    main()
