"""
How useful is the timetable model (v2 gradient boosting) in practice?

An AUC of 0.654 says the model ranks late flights above on-time ones more
often than not, but it doesn't tell an airline what it would get from using
it. This script answers three practical questions on the 2008 test flights:

1. Worked example: if an airline flagged the riskiest 10% of flights, what
   share of all late flights would it catch, and how many flagged flights
   would really be late? The same is shown for 5%, 20%, 30% and 50%.

2. Precision-recall curve: the full trade-off behind that example. Flag
   more flights and you catch more late ones (recall goes up), but a smaller
   share of the flagged flights are really late (precision goes down).

3. Calibration: if the model says "30% chance of delay" for a group of
   flights, are about 30% of them really late? (Same check as
   src/check_probabilities.py does for the 2-hours-before model.)

Same 30% sample and split as the README: train on 2004-2007, test on
Jan-Apr 2008.

Usage:
    python src/practical_value.py --db data/Fly.db [--pct 1] [--seed 42]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import check_probabilities
import model_v2
import model_v3

ROOT = Path(__file__).resolve().parents[1]

MODEL = "Gradient boosting (pre-departure)"
TOP_SHARES = [0.05, 0.10, 0.20, 0.30, 0.50]
EXAMPLE_SHARE = 0.10


def predict_test_risk(train, test):
    """Fit the timetable model on 2004-2007 and return its chance of delay for each 2008 flight."""
    build, num, cat = model_v2.SETUPS[MODEL]
    model = build(num, cat).fit(train[num + cat], train["Delayed"])
    return model.predict_proba(test[num + cat])[:, 1]


def flagging_table(late, risk, shares=TOP_SHARES):
    """For each share of flights flagged (riskiest first): how many are late, and how many late flights are caught."""
    rows = []
    for share in shares:
        # Reuse v3's 'riskiest 10%' rule so the numbers match the README's earlier tables
        result = model_v3.flag_riskiest(late, risk, share=share)
        rows.append({
            "flag_riskiest_share": share,
            "flagged_flights": result["flagged_flights"],
            "late_among_flagged": result["late_rate_in_flagged"],
            "times_the_average": result["times_the_average"],
            "share_of_late_flights_caught": result["share_of_late_flights_caught"],
        })
    return pd.DataFrame(rows)


def summary_table(late, risk):
    """Headline numbers: how well it ranks, how much better than guessing, and how honest its percentages are."""
    late, risk = np.asarray(late), np.asarray(risk)
    calibration = check_probabilities.calibration_table(late, risk)
    return pd.DataFrame([{
        "test_flights": len(late),
        "share_late": late.mean(),
        "average_predicted_chance": risk.mean(),
        "auc": roc_auc_score(late, risk),
        # Average precision: the area under the precision-recall curve.
        # Flagging flights at random would score the share late (about 0.25).
        "average_precision": average_precision_score(late, risk),
        "calibration_average_gap": check_probabilities.average_gap(calibration),
    }])


def plot_precision_recall(late, risk, ax=None, example_share=EXAMPLE_SHARE):
    """Precision-recall curve, with the 'flag the riskiest 10%' example marked on it."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    precision, recall, _ = precision_recall_curve(late, risk)
    ax.plot(recall * 100, precision * 100, color="#1f5fa8", label="Timetable model (v2 gradient boosting)")

    # Picking flights at random: every list is late at the overall rate, however long it is
    base_rate = np.mean(late) * 100
    ax.axhline(base_rate, color="#9aa0a6", linestyle="--", lw=1,
               label=f"Picking flights at random ({base_rate:.1f}% late)")

    # Mark the worked example so the chart and the README sentence point at the same spot
    example = model_v3.flag_riskiest(late, risk, share=example_share)
    x, y = example["share_of_late_flights_caught"] * 100, example["late_rate_in_flagged"] * 100
    ax.plot(x, y, "o", color="#d1495b", zorder=3)
    ax.annotate(f"Flag the riskiest {example_share:.0%}:\ncatches {x:.1f}% of late flights,\n"
                f"{y:.1f}% of flagged flights are late",
                xy=(x, y), xytext=(x + 8, y + 12), arrowprops={"arrowstyle": "-", "color": "#5f6368"})

    ax.set_xlabel("Share of all late flights caught (recall, %)")
    ax.set_ylabel("Share of flagged flights really late (precision, %)")
    ax.set_title("Flag more flights, catch more delays, but with more false alarms\n(test set: Jan-Apr 2008)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
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
    risk = predict_test_risk(train, test)
    late = test["Delayed"].values

    summary = summary_table(late, risk)
    summary["pct"], summary["seed"] = args.pct, args.seed
    print("\nSummary\n", summary.round(4).to_string(index=False))

    flagging = flagging_table(late, risk)
    print("\nFlagging the riskiest flights\n", flagging.round(4).to_string(index=False))
    example = flagging[flagging["flag_riskiest_share"] == EXAMPLE_SHARE].iloc[0]
    print(f"\nWorked example: flag the riskiest {EXAMPLE_SHARE:.0%} of flights and you catch "
          f"{example['share_of_late_flights_caught']:.1%} of all late flights; "
          f"{example['late_among_flagged']:.1%} of the flagged flights are late "
          f"(against {late.mean():.1%} of all flights)")

    calibration = check_probabilities.calibration_table(late, risk)
    print("\nPredicted chance vs reality\n", calibration.round(4).to_string())

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    summary.round(4).to_csv(out / "practical_value_v2.csv", index=False)
    flagging.round(4).to_csv(out / "flagging_v2.csv", index=False)
    calibration.round(4).to_csv(out / "calibration_v2.csv", index_label="predicted_chance_group")

    ax = plot_precision_recall(late, risk)
    ax.figure.savefig(ROOT / "images" / "precision_recall_v2.png", dpi=150)
    ax = check_probabilities.plot_calibration(calibration, label="Timetable model (v2 gradient boosting)")
    ax.figure.savefig(ROOT / "images" / "calibration_v2.png", dpi=150)
    print("\nSaved results/practical_value_v2.csv, results/flagging_v2.csv, results/calibration_v2.csv, "
          "images/precision_recall_v2.png and images/calibration_v2.png")


if __name__ == "__main__":
    main()
