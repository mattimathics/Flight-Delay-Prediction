"""
Save the model's predictions for every 2008 test flight, for the web page in docs/.

The web page (docs/index.html) runs no model. It reads docs/flights.json,
which this script writes: for each Jan-Apr 2008 test flight, its chance of
delay weeks ahead (timetable only) and 4, 2 and 1 hour before departure,
what the model knew 2 hours before, and what really happened. The models are
the same ones as in src/prediction_time.py, trained on 2004-2007 only, so
every flight on the page is one the model never saw.

Usage:
    python src/build_web_page.py --db data/Fly.db [--pct 1] [--seed 42]
"""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

import model_v2
import model_v3
import model_v4

ROOT = Path(__file__).resolve().parents[1]

# Short labels the page shows for each prediction time, with the cutoff in minutes
PREDICTION_TIMES = {"Weeks ahead": None, "4 hours before": 240, "2 hours before": 120, "1 hour before": 60}
EXPLAIN_AT = 120  # the page explains what the model knew 2 hours before

COLUMNS = ["date", "carrier", "flight_number", "origin", "dest", "scheduled_departure",
           "arrival_delay", "late", "risk_weeks", "risk_4h", "risk_2h", "risk_1h",
           "airport_late_share", "turnaround_minutes", "previous_flight_delay"]


def fit_and_predict(num, train, test):
    """Gradient boosting on 2004-2007; chance of delay for each 2008 flight, keyed by flight id."""
    model = model_v2.boosting(num, model_v4.CAT).fit(train[num + model_v4.CAT], train["Delayed"])
    risk = model.predict_proba(test[num + model_v4.CAT])[:, 1]
    return pd.Series(risk, index=test["FlightId"].values)


def predictions_at_each_time(df, db_path):
    """Chance of delay for every test flight at every prediction time, plus the 2-hour facts."""
    risks, facts = {}, None
    for label, cutoff in PREDICTION_TIMES.items():
        if cutoff is None:
            train, test = model_v2.split_by_year(df)
            risks[label] = fit_and_predict(model_v4.V3_NUM, train, test)
            continue
        at_cutoff = model_v4.add_same_day_features(df, db_path, cutoff)
        train, test = model_v2.split_by_year(at_cutoff)
        risks[label] = fit_and_predict(model_v4.V4_NUM, train, test)
        if cutoff == EXPLAIN_AT:
            facts = test.set_index("FlightId")[model_v4.AIRPORT_EARLIER + model_v4.PLANE_PREVIOUS]
    return pd.DataFrame(risks), facts


def names(db_path):
    """Airline names and airport cities, so the page can say 'Chicago' instead of 'ORD'."""
    with sqlite3.connect(db_path) as conn:
        carriers = pd.read_sql_query("SELECT Code, Description FROM carriers", conn)
        airports = pd.read_sql_query("SELECT iata, city, state FROM airports", conn)
    return (dict(zip(carriers["Code"], carriers["Description"])),
            {row.iata: f"{row.city}, {row.state}" for row in airports.itertuples()})


def summary(test, risks):
    """Headline numbers for the page: score and riskiest-10% result at each prediction time."""
    rows = []
    for label in risks.columns:
        risk = risks.loc[test["FlightId"], label].values
        flagged = model_v3.flag_riskiest(test["Delayed"], risk)
        rows.append({"prediction_time": label,
                     "auc": round(roc_auc_score(test["Delayed"], risk), 3),
                     "late_in_riskiest_10pct": round(flagged["late_rate_in_flagged"], 3)})
    return {"flights": len(test), "late_overall": round(test["Delayed"].mean(), 3), "by_time": rows}


def page_rows(test, risks, facts):
    """One short list per flight, in the order of COLUMNS, to keep the file small."""
    test = test.set_index("FlightId")
    table = pd.DataFrame({
        "date": [f"{y}-{int(m):02d}-{int(d):02d}" for y, m, d in zip(test["Year"], test["Month"], test["DayofMonth"])],
        "carrier": test["UniqueCarrier"],
        "flight_number": test["FlightNum"].astype(int),
        "origin": test["Origin"],
        "dest": test["Dest"],
        "scheduled_departure": test["CRSDepTime"].astype(int),
        "arrival_delay": test["ArrDelay"].astype(int),
        "late": test["Delayed"],
        "risk_weeks": risks.loc[test.index, "Weeks ahead"].round(3),
        "risk_4h": risks.loc[test.index, "4 hours before"].round(3),
        "risk_2h": risks.loc[test.index, "2 hours before"].round(3),
        "risk_1h": risks.loc[test.index, "1 hour before"].round(3),
        "airport_late_share": facts.loc[test.index, "AirportLateShareEarlier"].round(3),
        "turnaround_minutes": facts.loc[test.index, "PlaneTurnaroundMinutes"],
        "previous_flight_delay": facts.loc[test.index, "PlanePrevDepDelay"],
    }, index=test.index)
    table = table.sort_values(["origin", "date", "scheduled_departure"])
    # JSON has no NaN, so unknown values become null
    return [[None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row]
            for row in table[COLUMNS].itertuples(index=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = model_v3.load(args.db, args.pct, args.seed)
    _, test = model_v2.split_by_year(df)
    risks, facts = predictions_at_each_time(df, args.db)
    carriers, airports = names(args.db)

    rows = page_rows(test, risks, facts)
    used_airports = {code for row in rows for code in (row[3], row[4])}
    used_carriers = {row[1] for row in rows}
    page_data = {
        "summary": summary(test, risks),
        "columns": COLUMNS,
        "carriers": {code: carriers.get(code, code) for code in sorted(used_carriers)},
        "airports": {code: airports.get(code, code) for code in sorted(used_airports)},
        "flights": rows,
    }
    out = ROOT / "docs" / "flights.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(page_data, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(page_data["summary"], indent=1))
    print(f"Saved {len(rows):,} flights to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
