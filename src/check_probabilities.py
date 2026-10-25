"""
Can the 2-hours-before model's percentages be trusted, and when should it raise an alert?

1. Calibration: if the model says "30% chance of delay" for a group of
   flights, about 30% of them should really be late. This script groups the
   2008 test flights by predicted chance and compares each group's average
   prediction with the share that really arrived 15+ minutes late.

2. Alert level: an airline should act on a flight when the expected cost of
   doing nothing is bigger than the cost of acting:
       chance of delay x cost of a missed delay > cost of acting
   so it should flag a flight when
       chance of delay > cost of acting / cost of a missed delay.
   That rule only works if the chances are trustworthy, which is why step 1
   comes first. The two costs are inputs: the defaults (1 and 4) are an
   example, not real airline figures.

Usage:
    python src/check_probabilities.py --db data/Fly.db [--cost-of-acting 1] [--cost-of-missed-delay 4]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_v2
import model_v4

ROOT = Path(__file__).resolve().parents[1]

BEST_MODEL = "v4 gradient boosting (2 hours before)"
THRESHOLDS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def predict_test_risk(train, test):
    """Fit the 2-hours-before model on 2004-2007 and return its chance of delay for each 2008 flight."""
    build, num, cat = model_v4.SETUPS[BEST_MODEL]
    model = build(num, cat).fit(train[num + cat], train["Delayed"])
    return model.predict_proba(test[num + cat])[:, 1]


def calibration_table(late, risk, n_groups=10):
    """Group flights by predicted chance (0-10%, 10-20%, ...) and compare prediction with reality."""
    flights = pd.DataFrame({"late": np.asarray(late), "risk": np.asarray(risk)})
    edges = np.linspace(0, 1, n_groups + 1)
    flights["group"] = pd.cut(flights["risk"], edges, include_lowest=True)
    table = flights.groupby("group", observed=True).agg(
        flights=("late", "size"), predicted=("risk", "mean"), actually_late=("late", "mean"))
    table.index = [f"{int(g.left * 100)}-{int(round(g.right * 100))}%" for g in table.index]
    return table


def average_gap(table):
    """How far predictions are from reality, on average across all flights (in share points)."""
    gap = (table["predicted"] - table["actually_late"]).abs()
    return (gap * table["flights"]).sum() / table["flights"].sum()


def threshold_table(late, risk, thresholds=THRESHOLDS):
    """For each alert level: how many flights get flagged, and how good that list is."""
    late, risk = np.asarray(late), np.asarray(risk)
    rows = []
    for level in thresholds:
        flagged = risk >= level
        rows.append({
            "alert_if_chance_at_least": level,
            "share_of_flights_flagged": flagged.mean(),
            "late_among_flagged": late[flagged].mean() if flagged.any() else np.nan,
            "share_of_late_flights_caught": late[flagged].sum() / late.sum(),
        })
    return pd.DataFrame(rows)


def alert_level(cost_of_acting, cost_of_missed_delay):
    """Flag a flight once its chance of delay makes acting cheaper than waiting."""
    return cost_of_acting / cost_of_missed_delay


def plot_calibration(table, ax=None, label="2-hours-before model"):
    """Predicted chance vs share really late; a trustworthy model sits on the diagonal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 100], [0, 100], color="#9aa0a6", linestyle="--", lw=1, label="Perfectly trustworthy")
    ax.plot(table["predicted"] * 100, table["actually_late"] * 100, marker="o", color="#1f5fa8",
            label=label)
    ax.set_xlabel("Model's predicted chance of delay (%)")
    ax.set_ylabel("Share of those flights really late (%)")
    ax.set_title("Can the model's percentages be trusted?\n(test set: Jan-Apr 2008)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.figure.tight_layout()
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    ap.add_argument("--cost-of-acting", type=float, default=1, help="cost of acting on one flagged flight")
    ap.add_argument("--cost-of-missed-delay", type=float, default=4, help="cost of one delay nobody acted on")
    args = ap.parse_args()

    df = model_v4.load(args.db, args.pct, args.seed)
    train, test = model_v2.split_by_year(df)
    risk = predict_test_risk(train, test)
    late = test["Delayed"].values

    # Delays became more common every year, so a model trained on 2004-2007 may run a little low on 2008
    print("Share late by training year:", train.groupby("Year")["Delayed"].mean().round(3).to_dict())
    print(f"2008 test flights: {late.mean():.1%} late; model's average prediction {risk.mean():.1%}")

    calibration = calibration_table(late, risk)
    print("\nPredicted chance vs reality\n", calibration.round(3).to_string())
    print(f"Average gap between prediction and reality: {average_gap(calibration):.1%} points")

    thresholds = threshold_table(late, risk)
    print("\nAlert levels\n", thresholds.round(3).to_string(index=False))
    level = alert_level(args.cost_of_acting, args.cost_of_missed_delay)
    print(f"\nWith cost of acting {args.cost_of_acting:g} and cost of a missed delay "
          f"{args.cost_of_missed_delay:g}: flag flights with a chance of delay of {level:.0%} or more")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    calibration.round(4).to_csv(out / "calibration_v4.csv", index_label="predicted_chance_group")
    thresholds.round(4).to_csv(out / "alert_levels_v4.csv", index=False)
    ax = plot_calibration(calibration)
    ax.figure.savefig(ROOT / "images" / "calibration_v4.png", dpi=150)
    print("\nSaved results/calibration_v4.csv, results/alert_levels_v4.csv and images/calibration_v4.png")


if __name__ == "__main__":
    main()
