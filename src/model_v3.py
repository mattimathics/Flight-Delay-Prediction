"""
Flight delay prediction, v3: a stronger model that still only uses
information known before departure.

v2 (src/model_v2.py) uses the timetable, route, airline, date and plane age.
v3 adds three things an airline also knows in advance:

- OriginHourFlights: how many flights are scheduled to leave the same airport
  in the same hour of the same day (a busy hour means queues on the runway).
- DestHourFlights: how many flights are scheduled to land at the destination
  in the same hour of the same day.
- RoutePastDelayRate: the share of flights on the same route (origin to
  destination) that arrived 15+ minutes late in 2004-2007, the training years.

It also answers two questions a hiring manager or an airline would ask:

- Is the model better than simply looking up the route's past delay rate?
- If we flag the riskiest 10% of flights, how many of them are really late?

Same sample, same split as v2: train on 2004-2007, test on 2008 (Jan-Apr).

Usage:
    python src/model_v3.py --db data/Fly.db [--pct 1] [--seed 42]
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

import model_v2

ROOT = Path(__file__).resolve().parents[1]

TRAINING_YEARS = (2004, 2007)
# A route with only a handful of past flights is pulled towards the overall rate,
# as if it had this many extra "average" flights. See route_past_delay_rate().
ROUTE_SMOOTHING_FLIGHTS = 50
TOP_SHARE = 0.10

BUSYNESS = ["OriginHourFlights", "DestHourFlights"]
ROUTE_HISTORY = ["RoutePastDelayRate"]

V2_NUM = model_v2.PRE_DEPARTURE_NUM
V3_NUM = V2_NUM + BUSYNESS + ROUTE_HISTORY
CAT = model_v2.PRE_DEPARTURE_CAT

# Each setup is (model builder, number features, category features).
# The two "only" rows show how much each new idea adds on its own.
SETUPS = {
    "v2 gradient boosting (timetable, route, airline, plane age)": (model_v2.boosting, V2_NUM, CAT),
    "v3 gradient boosting + airport busyness only": (model_v2.boosting, V2_NUM + BUSYNESS, CAT),
    "v3 gradient boosting + route history only": (model_v2.boosting, V2_NUM + ROUTE_HISTORY, CAT),
    "v3 logistic regression (all new features)": (model_v2.logistic, V3_NUM, CAT),
    "v3 gradient boosting (all new features)": (model_v2.boosting, V3_NUM, CAT),
}
BENCHMARK = "Benchmark: route's past delay rate (no model)"

# Counts every scheduled flight in an (airport, day, hour) slot, cancelled ones
# included, because they were on the timetable when the forecast is made.
# CROSS JOIN makes SQLite read flightdetails once and look each row up in the
# small keys table, instead of scanning 31 million rows once per key.
SLOT_COUNT_QUERY = """
SELECT k.Year, k.Month, k.DayofMonth, k.Airport, k.Hour, COUNT(*) AS Flights
FROM flightdetails f
CROSS JOIN slot_keys k
WHERE f.Year = k.Year AND f.Month = k.Month AND f.DayofMonth = k.DayofMonth
  AND f.{airport} = k.Airport AND MIN(f.{time} / 100, 23) = k.Hour
GROUP BY k.Year, k.Month, k.DayofMonth, k.Airport, k.Hour
"""

ROUTE_HISTORY_QUERY = """
SELECT Origin, Dest, COUNT(*) AS Flights, SUM(ArrDelay >= 15) AS Late
FROM flightdetails
WHERE ArrDelay IS NOT NULL AND Year BETWEEN ? AND ?
GROUP BY Origin, Dest
"""

SLOT_KEYS = ["Year", "Month", "DayofMonth", "Airport", "Hour"]


def count_slot_flights(conn, df, airport_col, hour_col, time_col):
    """For each flight in df, count all flights scheduled at its airport in that hour that day."""
    flight_slots = pd.DataFrame({
        "Year": df["Year"].astype(int).values,
        "Month": df["Month"].astype(int).values,
        "DayofMonth": df["DayofMonth"].astype(int).values,
        "Airport": df[airport_col].values,
        "Hour": df[hour_col].astype(int).values,
    })
    # Only the slots our sampled flights fall in are needed, not every slot in the database
    keys = flight_slots.drop_duplicates()
    conn.execute("DROP TABLE IF EXISTS temp.slot_keys")
    conn.execute("CREATE TEMP TABLE slot_keys (Year INTEGER, Month INTEGER, DayofMonth INTEGER, "
                 "Airport TEXT, Hour INTEGER, PRIMARY KEY (Year, Month, DayofMonth, Airport, Hour))")
    conn.executemany("INSERT INTO slot_keys VALUES (?, ?, ?, ?, ?)", keys.itertuples(index=False))
    counts = pd.read_sql_query(SLOT_COUNT_QUERY.format(airport=airport_col, time=time_col), conn)

    # Line each flight up with its slot's count, keeping df's row order
    return flight_slots.merge(counts, on=SLOT_KEYS, how="left")["Flights"].values


def add_airport_busyness(df, db_path):
    """Add how many flights are scheduled at the origin and destination in the flight's hour."""
    with sqlite3.connect(db_path) as conn:
        df["OriginHourFlights"] = count_slot_flights(conn, df, "Origin", "CRSDepHour", "CRSDepTime")
        df["DestHourFlights"] = count_slot_flights(conn, df, "Dest", "CRSArrHour", "CRSArrTime")
    return df


def route_past_delay_rate(db_path, first_year=TRAINING_YEARS[0], last_year=TRAINING_YEARS[1]):
    """Share of each route's flights that were 15+ min late, using the training years only."""
    with sqlite3.connect(db_path) as conn:
        routes = pd.read_sql_query(ROUTE_HISTORY_QUERY, conn, params=(first_year, last_year))
    overall_rate = routes["Late"].sum() / routes["Flights"].sum()
    # A route flown 3 times with 2 late flights should not get a 67% rate. Adding
    # ROUTE_SMOOTHING_FLIGHTS imaginary average flights pulls small routes to the mean
    # while barely moving busy routes that have thousands of flights.
    routes["RoutePastDelayRate"] = ((routes["Late"] + ROUTE_SMOOTHING_FLIGHTS * overall_rate)
                                    / (routes["Flights"] + ROUTE_SMOOTHING_FLIGHTS))
    return routes[["Origin", "Dest", "RoutePastDelayRate"]], overall_rate


def add_route_history(df, db_path):
    """Add each flight's route past delay rate; routes never flown before get the overall rate."""
    routes, overall_rate = route_past_delay_rate(db_path)
    df = df.merge(routes, on=["Origin", "Dest"], how="left")
    df["RoutePastDelayRate"] = df["RoutePastDelayRate"].fillna(overall_rate)
    return df


def flag_riskiest(y_true, risk, share=TOP_SHARE):
    """Flag the share of flights with the highest risk and say how well that list works."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    risk = pd.Series(risk).reset_index(drop=True)
    n_flagged = int(round(len(risk) * share))
    # Ties are broken by row order, so the same scores always flag the same flights
    flagged = risk.sort_values(ascending=False, kind="stable").index[:n_flagged]
    late_rate_flagged = y_true[flagged].mean()
    return {
        "flagged_flights": n_flagged,
        "late_rate_in_flagged": late_rate_flagged,
        "late_rate_overall": y_true.mean(),
        "times_the_average": late_rate_flagged / y_true.mean(),
        "share_of_late_flights_caught": y_true[flagged].sum() / y_true.sum(),
    }


def score(name, y_true, risk):
    """One results row: AUC plus the 'riskiest 10%' numbers."""
    return {"model": name, "test_auc": roc_auc_score(y_true, risk), **flag_riskiest(y_true, risk)}


def fit_and_score(train, test):
    """Score the no-model benchmark, then fit and score every setup in SETUPS."""
    results = [score(BENCHMARK, test["Delayed"], test["RoutePastDelayRate"])]
    fitted = {}
    for name, (build, num, cat) in SETUPS.items():
        model = build(num, cat).fit(train[num + cat], train["Delayed"])
        risk = model.predict_proba(test[num + cat])[:, 1]
        results.append(score(name, test["Delayed"], risk))
        fitted[name] = (model, num + cat)
    table = pd.DataFrame(results)
    table["n_train"], table["n_test"] = len(train), len(test)
    return table, fitted


def plot_riskiest(results, ax=None):
    """Bar chart: share of flights late, on average vs in each method's riskiest 10%."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    v2_name, v3_name = list(SETUPS)[0], list(SETUPS)[-1]
    rows = results.set_index("model")
    bars = pd.Series({
        "All 2008 flights": rows.loc[v3_name, "late_rate_overall"],
        "Riskiest 10%: route's\npast delay rate": rows.loc[BENCHMARK, "late_rate_in_flagged"],
        "Riskiest 10%: v2 model": rows.loc[v2_name, "late_rate_in_flagged"],
        "Riskiest 10%: v3 model": rows.loc[v3_name, "late_rate_in_flagged"],
    })
    colors = ["#9aa0a6", "#9aa0a6", "#7ba7d9", "#1f5fa8"]
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
    """v2's sample plus the three new pre-departure features."""
    df = model_v2.load(db_path, pct, seed)
    df = add_airport_busyness(df, db_path)
    return add_route_history(df, db_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = load(args.db, args.pct, args.seed)
    train, test = model_v2.split_by_year(df)
    print(f"Train {len(train):,} rows | Test {len(test):,} rows")

    results, _ = fit_and_score(train, test)
    results["pct"], results["seed"] = args.pct, args.seed
    pd.set_option("display.width", 200)
    print(results[["model", "test_auc", "late_rate_in_flagged", "times_the_average",
                   "share_of_late_flights_caught"]].round(3).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.round(4).to_csv(out / "model_comparison_v3.csv", index=False)
    ax = plot_riskiest(results)
    ax.figure.savefig(ROOT / "images" / "riskiest_flights_v3.png", dpi=150)
    print(f"\nSaved {out / 'model_comparison_v3.csv'} and images/riskiest_flights_v3.png")


if __name__ == "__main__":
    main()
