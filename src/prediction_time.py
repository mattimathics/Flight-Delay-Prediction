"""
How does accuracy change as the prediction is made closer to departure?

The same 2-hours-before model (src/model_v4.py) is rebuilt with its cutoff
moved to 4 hours, 2 hours and 1 hour before scheduled departure. At each
cutoff it only sees what had happened by then, so later cutoffs know more.
The timetable-only model stands for "weeks ahead", and the leaky model from
v2 (which is given the flight's own departure delay) shows the ceiling that
can only be reached after take-off.

Same sample and split as v2-v4: train on 2004-2007, test on 2008 (Jan-Apr).

Usage:
    python src/prediction_time.py --db data/Fly.db [--pct 1] [--seed 42]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import model_v2
import model_v3
import model_v4

ROOT = Path(__file__).resolve().parents[1]

CUTOFFS_MINUTES = [240, 120, 60]
LEAKY_NUM = model_v4.V3_NUM + model_v2.LEAKY_NUM


def score_model(name, num, train, test):
    """Fit gradient boosting on the given number features and score it on 2008."""
    model = model_v2.boosting(num, model_v4.CAT).fit(train[num + model_v4.CAT], train["Delayed"])
    risk = model.predict_proba(test[num + model_v4.CAT])[:, 1]
    return model_v3.score(name, test["Delayed"], risk)


def compare_prediction_times(df, db_path, cutoffs=CUTOFFS_MINUTES):
    """One results row per prediction time, from weeks ahead to after take-off."""
    train, test = model_v2.split_by_year(df)
    rows = [{"prediction_time": "Weeks ahead (timetable only)",
             **score_model("timetable", model_v4.V3_NUM, train, test)}]
    for cutoff in cutoffs:
        # Rebuild the same-day features as they would look at this cutoff
        at_cutoff = model_v4.add_same_day_features(df, db_path, cutoff)
        train, test = model_v2.split_by_year(at_cutoff)
        row = score_model(f"{cutoff} min before", model_v4.V4_NUM, train, test)
        row["prev_flight_delay_known"] = test["PlanePrevDepDelay"].notna().mean()
        rows.append({"prediction_time": f"{cutoff // 60} hour{'s' if cutoff > 60 else ''} before", **row})
    train, test = model_v2.split_by_year(df)
    rows.append({"prediction_time": "After take-off (leaky)",
                 **score_model("leaky", LEAKY_NUM, train, test)})
    table = pd.DataFrame(rows).drop(columns="model")
    table["n_train"], table["n_test"] = len(train), len(test)
    return table


def plot_prediction_time(table, ax=None):
    """Score at each prediction time, with the leaky model as a dashed line it should never reach."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    honest = table[table["prediction_time"] != "After take-off (leaky)"]
    leaky_auc = table.loc[table["prediction_time"] == "After take-off (leaky)", "test_auc"].item()
    ax.plot(honest["prediction_time"], honest["test_auc"], marker="o", color="#1f5fa8")
    for x, value in enumerate(honest["test_auc"]):
        ax.text(x, value + 0.012, f"{value:.3f}", ha="center")
    ax.axhline(leaky_auc, color="#c0504d", linestyle="--", lw=1)
    ax.text(0, leaky_auc - 0.03, f"Leaky model, uses the flight's own delay ({leaky_auc:.3f})", color="#c0504d")
    ax.axhline(0.5, color="#9aa0a6", linestyle=":", lw=1)
    ax.text(0, 0.51, "Coin flip (0.5)", color="#6b6f73")
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Score on 2008 flights (AUC)")
    ax.set_title("Predictions get better closer to departure, without leakage")
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.tight_layout()
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = model_v3.load(args.db, args.pct, args.seed)
    table = compare_prediction_times(df, args.db)
    table["pct"], table["seed"] = args.pct, args.seed
    pd.set_option("display.width", 200)
    print(table[["prediction_time", "test_auc", "late_rate_in_flagged", "times_the_average",
                 "share_of_late_flights_caught", "prev_flight_delay_known"]].round(3).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    table.round(4).to_csv(out / "prediction_time.csv", index=False)
    ax = plot_prediction_time(table)
    ax.figure.savefig(ROOT / "images" / "prediction_time.png", dpi=150)
    print("\nSaved results/prediction_time.csv and images/prediction_time.png")


if __name__ == "__main__":
    main()
