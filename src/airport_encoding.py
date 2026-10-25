"""
How should the model see airports? A comparison of two ways.

Gradient boosting in scikit-learn can handle at most 255 categories in one
column. There are 321 origin airports in 2004-2007, so model_v2.boosting()
keeps the 254 busiest as their own category and puts the rest in one shared
"other" group.

The alternative is target encoding: replace each airport's name with a number,
the share of its flights that arrived 15+ minutes late in the training years
(2004-2007). No airport is lumped in with others, and the model gets a number
it can split on directly.

This script trains the timetable-only model and the 2-hours-before model both
ways, on the same sample and split as model_v4.py, and compares them on the
2008 test flights.

Usage:
    python src/airport_encoding.py --db data/Fly.db [--pct 1] [--seed 42]
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

import model_v2
import model_v3
import model_v4

ROOT = Path(__file__).resolve().parents[1]

AIRPORT_COLUMNS = ["Origin", "Dest"]
AIRPORT_RATES = ["OriginPastDelayRate", "DestPastDelayRate"]

CAT = model_v2.PRE_DEPARTURE_CAT
CAT_WITHOUT_AIRPORTS = [col for col in CAT if col not in AIRPORT_COLUMNS]

# The same features as the published models, with the airports given either as
# categories (current) or as their past delay rates (target encoding)
SETUPS = {
    "Timetable only: airports as categories (current)":
        (model_v2.boosting, model_v4.V3_NUM, CAT),
    "Timetable only: airports as past delay rates":
        (model_v2.boosting, model_v4.V3_NUM + AIRPORT_RATES, CAT_WITHOUT_AIRPORTS),
    "2 hours before: airports as categories (current)":
        (model_v2.boosting, model_v4.V4_NUM, CAT),
    "2 hours before: airports as past delay rates":
        (model_v2.boosting, model_v4.V4_NUM + AIRPORT_RATES, CAT_WITHOUT_AIRPORTS),
}

# {airport} is Origin or Dest; the two ? are the first and last training year
AIRPORT_HISTORY_QUERY = """
SELECT {airport}, COUNT(*) AS Flights, SUM(ArrDelay >= 15) AS Late
FROM flightdetails
WHERE ArrDelay IS NOT NULL AND Year BETWEEN ? AND ?
GROUP BY {airport}
"""


def airport_past_delay_rate(db_path, airport_col, first_year=model_v3.TRAINING_YEARS[0],
                            last_year=model_v3.TRAINING_YEARS[1]):
    """Share of each airport's flights that were 15+ min late, using the training years only."""
    query = AIRPORT_HISTORY_QUERY.format(airport=airport_col)
    with sqlite3.connect(db_path) as conn:
        airports = pd.read_sql_query(query, conn, params=(first_year, last_year))
    overall_rate = airports["Late"].sum() / airports["Flights"].sum()
    # Same smoothing as the route rate in model_v3: a tiny airport is pulled
    # towards the average, as if it also had 50 average flights
    smoothing = model_v3.ROUTE_SMOOTHING_FLIGHTS
    rate_col = f"{airport_col}PastDelayRate"
    airports[rate_col] = (airports["Late"] + smoothing * overall_rate) / (airports["Flights"] + smoothing)
    return airports[[airport_col, rate_col]], overall_rate


def add_airport_rates(df, db_path):
    """Add each flight's origin and destination past delay rate; new airports get the overall rate."""
    for airport_col in AIRPORT_COLUMNS:
        airports, overall_rate = airport_past_delay_rate(db_path, airport_col)
        df = df.merge(airports, on=airport_col, how="left")
        rate_col = f"{airport_col}PastDelayRate"
        df[rate_col] = df[rate_col].fillna(overall_rate)
    return df


def airports_over_the_cap(train, max_categories=255):
    """How many airports, and what share of training flights, end up in the shared 'other' group."""
    # The encoder keeps max_categories - 1 airports and uses the last slot for "other"
    kept = max_categories - 1
    summary = {}
    for airport_col in AIRPORT_COLUMNS:
        counts = train[airport_col].value_counts()
        summary[airport_col] = {
            "airports_in_training": len(counts),
            "airports_grouped_as_other": max(len(counts) - kept, 0),
            "share_of_training_flights_grouped": counts.iloc[kept:].sum() / counts.sum(),
        }
    return pd.DataFrame(summary).T


def load(db_path, pct, seed):
    """model_v4's sample and features plus each airport's past delay rate."""
    df = model_v4.load(db_path, pct, seed)
    return add_airport_rates(df, db_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = load(args.db, args.pct, args.seed)
    train, test = model_v2.split_by_year(df)
    print(f"Train {len(train):,} rows | Test {len(test):,} rows")

    cap = airports_over_the_cap(train)
    print("\nAirports that the 255-category cap groups together\n", cap.round(4).to_string())

    results, _ = model_v3.fit_and_score(train, test, SETUPS)
    results["pct"], results["seed"] = args.pct, args.seed
    pd.set_option("display.width", 200)
    print()
    print(results[["model", "test_auc", "late_rate_in_flagged"]].round(4).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.round(4).to_csv(out / "airport_encoding.csv", index=False)
    cap.round(4).to_csv(out / "airport_category_cap.csv")
    print(f"\nSaved results/airport_encoding.csv and results/airport_category_cap.csv")


if __name__ == "__main__":
    main()
