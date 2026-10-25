"""
What drives the timetable model's predictions?

Permutation importance: take the 2008 test flights, shuffle one column (say,
the airline) so it no longer matches its flight, and see how much the
model's AUC drops. A big drop means the model leans on that column; no drop
means it could do without it. The model is never retrained, so this
measures what the fitted model uses, not what causes delays.

Each column is shuffled 5 times and the drops are averaged, so one lucky
shuffle can't decide the ranking.

One catch: when two columns carry the same information (scheduled departure
and arrival hour, or distance and scheduled flight length), shuffling one
leaves the other in place, so each can look less important than the pair
really is.

Same model, 30% sample and split as the README: v2 gradient boosting, train
on 2004-2007, test on Jan-Apr 2008.

Usage:
    python src/what_drives_delays.py --db data/Fly.db [--pct 1] [--seed 42]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

import model_v2

ROOT = Path(__file__).resolve().parents[1]

MODEL = "Gradient boosting (pre-departure)"
N_REPEATS = 5

PLAIN_NAMES = {
    "CRSDepHour": "Scheduled departure hour",
    "CRSArrHour": "Scheduled arrival hour",
    "CRSElapsedTime": "Scheduled flight length",
    "Distance": "Distance",
    "PlaneAge": "Plane age",
    "Month": "Month",
    "DayOfWeek": "Day of the week",
    "UniqueCarrier": "Airline",
    "Origin": "Departure airport",
    "Dest": "Arrival airport",
}


def fit_model(train):
    """Fit the timetable model on 2004-2007; return it with the columns it uses."""
    build, num, cat = model_v2.SETUPS[MODEL]
    columns = num + cat
    return build(num, cat).fit(train[columns], train["Delayed"]), columns


def importance_table(model, test, columns, n_repeats=N_REPEATS, seed=0):
    """AUC drop when each column is shuffled, biggest first."""
    # A fixed seed makes the shuffles, and so the table, the same on every run
    result = permutation_importance(model, test[columns], test["Delayed"], scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=seed)
    table = pd.DataFrame({
        "feature": columns,
        "plain_name": [PLAIN_NAMES.get(column, column) for column in columns],
        "auc_drop": result.importances_mean,
        "auc_drop_spread": result.importances_std,
    })
    return table.sort_values("auc_drop", ascending=False, ignore_index=True)


def plot_importance(table, ax=None):
    """Horizontal bars: how much the score drops when each column is shuffled."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ordered = table.iloc[::-1]  # biggest bar at the top
    ax.barh(ordered["plain_name"], ordered["auc_drop"], xerr=ordered["auc_drop_spread"],
            color="#1f5fa8", ecolor="#5f6368")
    for y, value in enumerate(ordered["auc_drop"]):
        ax.text(value + table["auc_drop"].max() * 0.02, y, f"{value:.3f}", va="center")
    ax.set_xlabel("Drop in AUC when this column is shuffled")
    ax.set_title("What the timetable model relies on most\n(test set: Jan-Apr 2008)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, table["auc_drop"].max() * 1.2)
    ax.figure.tight_layout()
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = model_v2.load(args.db, args.pct, args.seed)
    train, test = model_v2.split_by_year(df)
    print(f"Train {len(train):,} flights | Test {len(test):,} flights (pct={args.pct}, seed={args.seed})")

    model, columns = fit_model(train)
    table = importance_table(model, test, columns)
    print("\nDrop in AUC when each column is shuffled\n", table.round(4).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    table.round(4).to_csv(out / "feature_importance_v2.csv", index=False)
    ax = plot_importance(table)
    ax.figure.savefig(ROOT / "images" / "feature_importance_v2.png", dpi=150)
    print("\nSaved results/feature_importance_v2.csv and images/feature_importance_v2.png")


if __name__ == "__main__":
    main()
