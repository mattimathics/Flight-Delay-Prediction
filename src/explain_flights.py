"""
Why was this flight flagged? Reasons for each flight on the web page.

The 2-hours-before model (the same one as src/build_web_page.py) is retrained on
2004-2007. For each flight on the page, each group of facts the model uses (the plane
flying in, the airport's delays so far today, the time of day, ...) is swapped for the
facts of ordinary training flights, and the flight is rescored. How far the chance of
delay moves is how much that group pushed this flight's chance up or down.

By default each push is an exact Shapley value (what SHAP estimates): the group's effect
averaged over every mix of the other groups being this flight's or ordinary, so the pushes
add up exactly to the gap between this flight and an ordinary one. --method one-at-a-time
swaps each group alone instead, which is faster but doesn't add up. Either way it only
calls the model's own predict_proba, so no extra package is needed. The page shows the
biggest three.

Writes:
  docs/flights.json             gets a "reasons_2h" column: up to three [group, points]
  results/flight_reasons.csv    how often each group is a flight's biggest reason

Usage:
    python src/explain_flights.py --db data/Fly.db --pct 30 --show-pct 1
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_web_page
import model_v2
import model_v3
import model_v4

ROOT = Path(__file__).resolve().parents[1]
SAVED_MODEL = ROOT / "data" / "model_2h.joblib"

# Every column the 2-hour model uses, in groups a traveller would recognise.
# The keys are what the page stores; docs/index.html turns them into words.
GROUPS = {
    "plane": model_v4.PLANE_PREVIOUS,                       # the plane flying in
    "airport_today": model_v4.AIRPORT_EARLIER,              # departure airport's delays so far today
    "time": ["CRSDepHour", "CRSArrHour"],                   # time of day
    "date": ["Month", "DayOfWeek"],                         # month and day of the week
    "airline": ["UniqueCarrier"],
    "route": ["Origin", "Dest", "RoutePastDelayRate"],      # the airports and the route's track record
    "busyness": model_v3.BUSYNESS,                          # flights scheduled in the same hour
    "length": ["CRSElapsedTime", "Distance"],
    "plane_age": ["PlaneAge"],
}
FEATURES = model_v4.V4_NUM + model_v4.CAT
REFERENCES = 20  # ordinary training flights each flight is compared with
TOP = 3          # reasons shown per flight


def check_groups():
    """Every model column must sit in exactly one group, or a reason would be missing or doubled."""
    grouped = [col for cols in GROUPS.values() for col in cols]
    assert sorted(grouped) == sorted(FEATURES), "GROUPS must cover each model column once"


def train_2h_model(df, db_path):
    """The 2-hours-before model and the data it was trained and tested on."""
    at_cutoff = model_v4.add_same_day_features(df, db_path, 120)
    train, test = model_v2.split_by_year(at_cutoff)
    model = model_v2.boosting(model_v4.V4_NUM, model_v4.CAT).fit(train[FEATURES], train["Delayed"])
    return model, train, test


def group_pushes(model, flights, references):
    """Points (0-100) each group moves each flight's chance, against ordinary flights' values."""
    flights = flights[FEATURES].reset_index(drop=True)
    actual = model.predict_proba(flights)[:, 1]
    # Stack one copy of every flight per reference flight, so each group needs one predict call
    repeated = flights.loc[flights.index.repeat(len(references))].reset_index(drop=True)
    tiled = pd.concat([references[FEATURES]] * len(flights), ignore_index=True)
    pushes = {}
    for group, cols in GROUPS.items():
        swapped = repeated.copy()
        swapped[cols] = tiled[cols].values
        # The chance this flight would have with an ordinary flight's values for this group
        ordinary = model.predict_proba(swapped)[:, 1].reshape(len(flights), len(references)).mean(axis=1)
        pushes[group] = (actual - ordinary) * 100
    return pd.DataFrame(pushes)


def shapley_pushes(model, flights, references):
    """Exact Shapley values per group (src/check_reasons.py): they add up to the gap from ordinary flights."""
    import check_reasons  # imported here because check_reasons imports this module
    masks = check_reasons.coalitions(len(GROUPS))
    rows, baseline = [], None
    for _, flight in flights.reset_index(drop=True).iterrows():
        values = check_reasons.coalition_values(model, flight, references, masks)
        rows.append(check_reasons.shapley(values, masks))
        baseline = values[0]  # every group ordinary: the same for every flight
    return pd.DataFrame(rows, columns=list(GROUPS)), baseline


def top_reasons(pushes, top=TOP, min_points=1):
    """Each flight's biggest pushes, largest first, as [group, whole points]; tiny ones are dropped."""
    reasons = []
    for _, row in pushes.iterrows():
        biggest = row.reindex(row.abs().sort_values(ascending=False).index)[:top]
        reasons.append([[group, int(round(points))] for group, points in biggest.items()
                        if abs(points) >= min_points])
    return reasons


def reason_summary(pushes):
    """How often each group is a flight's biggest reason, and its typical size."""
    biggest = pushes.abs().idxmax(axis=1)
    table = pd.DataFrame({
        "group": list(GROUPS),
        "share_biggest_reason": [(biggest == g).mean() for g in GROUPS],
        "mean_points_up_or_down": [pushes[g].abs().mean() for g in GROUPS],
        "share_raises_chance": [(pushes[g] > 0).mean() for g in GROUPS],
    })
    return table.sort_values("share_biggest_reason", ascending=False)


def page_key(flights):
    """The fields that identify a flight in docs/flights.json."""
    return list(zip(
        [f"{y}-{int(m):02d}-{int(d):02d}" for y, m, d in zip(flights["Year"], flights["Month"], flights["DayofMonth"])],
        flights["UniqueCarrier"], flights["FlightNum"].astype(int), flights["Origin"], flights["Dest"],
        flights["CRSDepTime"].astype(int)))


def set_column(page, keys, values, name):
    """Set one column of the page data, matching each flight by its identifying fields; adds it if new."""
    by_key = dict(zip(keys, values))
    cols = page["columns"]
    at = [cols.index(c) for c in ("date", "carrier", "flight_number", "origin", "dest", "scheduled_departure")]
    if name not in cols:
        cols.append(name)
        for row in page["flights"]:
            row.append(None)
    where = cols.index(name)
    matched = 0
    for row in page["flights"]:
        key = tuple(row[i] for i in at)
        if key in by_key:
            row[where] = by_key[key]
            matched += 1
    return matched


def add_to_page(page, keys, reasons):
    """Add a reasons_2h column to the page data."""
    return set_column(page, keys, reasons, "reasons_2h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    ap.add_argument("--show-pct", type=float, default=None, help="the --show-pct the page was built with")
    ap.add_argument("--method", choices=["shapley", "one-at-a-time"], default="shapley",
                    help="exact Shapley values (default; about 50 minutes for 23,000 flights) or one group at a time")
    ap.add_argument("--chances-only", action="store_true",
                    help="only update the page's 2-hour chances from the model, keeping its reasons")
    ap.add_argument("--from-saved", action="store_true", help=f"reuse {SAVED_MODEL.relative_to(ROOT)}")
    args = ap.parse_args()
    check_groups()

    if args.from_saved:
        model, references, test = joblib.load(SAVED_MODEL)
    else:
        model, train, test = train_2h_model(model_v3.load(args.db, args.pct, args.seed), args.db)
        # Ordinary flights to compare with: a fixed random draw from the training years
        references = train.sample(REFERENCES, random_state=args.seed)
        # Training takes hours at 30%, so keep what a rerun needs (not the 8.5 million training rows)
        joblib.dump((model, references, test), SAVED_MODEL)

    shown = build_web_page.flights_to_show(test, args.show_pct, args.seed)
    page_file = ROOT / "docs" / "flights.json"
    page = json.loads(page_file.read_text(encoding="utf-8"))
    # The page's 2-hour chance must come from the same model as its reasons, or they won't add up
    chances = [round(float(c), 3) for c in model.predict_proba(shown[FEATURES])[:, 1]]
    matched = set_column(page, page_key(shown), chances, "risk_2h")
    print(f"Updated the 2-hour chance for {matched:,} of {len(page['flights']):,} page flights")
    if args.chances_only:
        page_file.write_text(json.dumps(page, separators=(",", ":")), encoding="utf-8")
        return

    if args.method == "shapley":
        # Shapley values add up exactly, so the page can say how far this flight is from an ordinary one
        pushes, baseline = shapley_pushes(model, shown, references)
        print(f"An ordinary flight's chance (all groups ordinary): {baseline:.1f}%")
    else:
        pushes, baseline = group_pushes(model, shown, references), None

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    summary = reason_summary(pushes)
    summary.round(4).to_csv(out / "flight_reasons.csv", index=False)
    print(summary.round(3).to_string(index=False))

    matched = add_to_page(page, page_key(shown), top_reasons(pushes))
    # Tell the page which method made the reasons, and the ordinary flight's chance they start from
    page["summary"]["reasons_method"] = args.method
    page["summary"]["ordinary_chance_2h"] = None if baseline is None else round(baseline / 100, 3)
    page_file.write_text(json.dumps(page, separators=(",", ":")), encoding="utf-8")
    print(f"\nAdded reasons for {matched:,} of {len(page['flights']):,} page flights to {page_file}")


if __name__ == "__main__":
    main()
