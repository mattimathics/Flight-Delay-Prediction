"""
How sure are the headline scores? Error bars from a bootstrap.

Each score in the README comes from one test set: 695,298 flights from Jan-Apr 2008.
A different set of flights would give a slightly different score. To see how much,
the test flights are resampled with replacement 1,000 times (a bootstrap) and every
model is rescored on each copy. The middle 95% of those scores is the error bar.

The same resampled copies are used for every model, so the gap between two models
also gets an error bar ("is the 2-hour model really better than weeks ahead?").

The models are the same ones as in src/build_web_page.py and src/prediction_time.py,
trained on 2004-2007 only. Their predictions for every test flight are saved to
data/test_predictions.csv.gz, so a rerun with --from-saved skips the hours of training.

Usage:
    python src/error_bars.py --db data/Fly.db --weather data/weather --pct 30
    python src/error_bars.py --from-saved   # reuse the saved predictions
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import build_web_page
import model_v2
import model_v3

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "data" / "test_predictions.csv.gz"

# Column in the predictions table -> name in the results table
MODELS = {
    "benchmark": "Benchmark: route's past delay rate (no model)",
    "Weeks ahead": "Weeks ahead (timetable only)",
    "4 hours before": "4 hours before",
    "2 hours before": "2 hours before",
    build_web_page.WITH_WEATHER: "2 hours before + weather",
    "1 hour before": "1 hour before",
}

# The comparisons the README's story rests on: (better model, model it should beat)
COMPARISONS = [
    ("Weeks ahead", "benchmark"),
    ("2 hours before", "Weeks ahead"),
    ("2 hours before", "4 hours before"),
    ("1 hour before", "2 hours before"),
    (build_web_page.WITH_WEATHER, "2 hours before"),
]


def test_predictions(df, db_path, weather_folder=None):
    """Every model's chance of delay for every 2008 test flight, plus whether it was late."""
    # The same training as the web page, so these are exactly the page's models
    risks, _ = build_web_page.predictions_at_each_time(df, db_path, weather_folder)
    _, test = model_v2.split_by_year(df)
    test = test.set_index("FlightId")
    table = risks.loc[test.index].copy()
    # The benchmark is not a model: each route's delay rate in 2004-2007 (model_v3.py)
    table.insert(0, "benchmark", test["RoutePastDelayRate"])
    table.insert(0, "late", test["Delayed"].astype(int))
    return table


class Scorer:
    """Scores one model on a resampled test set fast, by sorting its risks only once."""

    def __init__(self, late, risk):
        # Sort once from lowest to highest risk; each resample only changes how often a flight counts
        self.order = np.argsort(risk, kind="stable")
        sorted_risk = risk[self.order]
        self.late = late[self.order].astype(float)
        # Flights with the same risk form one group, so ties count as half right (as in the usual AUC)
        self.group_starts = np.flatnonzero(np.r_[True, sorted_risk[1:] != sorted_risk[:-1]])

    def auc(self, counts):
        """AUC when flight i appears counts[i] times: chance a late flight is rated above an on-time one."""
        weights = counts[self.order].astype(float)
        late = np.add.reduceat(weights * self.late, self.group_starts)
        on_time = np.add.reduceat(weights * (1 - self.late), self.group_starts)
        # On-time flights rated below each group, plus half of those tied with it
        below = np.cumsum(on_time) - on_time
        return float((late * (below + on_time / 2)).sum() / (late.sum() * on_time.sum()))

    def late_rate_in_riskiest(self, counts, share=model_v3.TOP_SHARE):
        """Share late among the riskiest 10% of the resampled flights (the warning list)."""
        weights = counts[self.order][::-1].astype(float)  # riskiest first
        late = self.late[::-1]
        n_flagged = round(weights.sum() * share)
        # Take whole flights until the list is full; the last one may count only partly
        taken = np.minimum(weights, np.maximum(n_flagged - (np.cumsum(weights) - weights), 0))
        return float((taken * late).sum() / n_flagged)


def bootstrap(table, n_resamples=1000, seed=42):
    """Score every model on each resampled copy of the test set; one row per copy."""
    late = table["late"].to_numpy()
    scorers = {col: Scorer(late, table[col].to_numpy()) for col in MODELS if col in table}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_resamples):
        # Draw as many flights as the test set has, with replacement: counts[i] = times flight i is drawn
        counts = np.bincount(rng.integers(0, len(table), len(table)), minlength=len(table))
        row = {}
        for col, scorer in scorers.items():
            row[(col, "auc")] = scorer.auc(counts)
            row[(col, "late")] = scorer.late_rate_in_riskiest(counts)
        rows.append(row)
    return pd.DataFrame(rows)


def interval(values):
    """The middle 95% of the resampled scores."""
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def summarise(table, draws):
    """Each model's score on the real test set, with its 95% error bar."""
    late = table["late"].to_numpy()
    everyone = np.ones(len(table), dtype=int)  # the real test set: every flight counted once
    rows = []
    for col, name in MODELS.items():
        if col not in table:
            continue
        scorer = Scorer(late, table[col].to_numpy())
        auc_low, auc_high = interval(draws[(col, "auc")])
        late_low, late_high = interval(draws[(col, "late")])
        rows.append({"model": name,
                     "test_auc": scorer.auc(everyone), "auc_low": auc_low, "auc_high": auc_high,
                     "late_rate_in_flagged": scorer.late_rate_in_riskiest(everyone),
                     "late_low": late_low, "late_high": late_high})
    out = pd.DataFrame(rows)
    out["n_test"], out["n_resamples"] = len(table), len(draws)
    return out


def compare(table, draws):
    """How much better one model is than another, with a 95% error bar on the gap."""
    late = table["late"].to_numpy()
    everyone = np.ones(len(table), dtype=int)
    rows = []
    for better, baseline in COMPARISONS:
        if better not in table or baseline not in table:
            continue
        gap = Scorer(late, table[better].to_numpy()).auc(everyone) - Scorer(late, table[baseline].to_numpy()).auc(everyone)
        # Same resampled copies for both models, so the gap's error bar is much tighter than either score's
        low, high = interval(draws[(better, "auc")] - draws[(baseline, "auc")])
        rows.append({"comparison": f"{MODELS[better]} vs {MODELS[baseline]}",
                     "auc_gap": gap, "gap_low": low, "gap_high": high,
                     "share_of_resamples_better": float((draws[(better, "auc")] > draws[(baseline, "auc")]).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--weather", default=None, help="folder of weather reports (src/download_weather.py)")
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--from-saved", action="store_true", help=f"reuse {SAVED.relative_to(ROOT)} instead of training")
    args = ap.parse_args()

    if args.from_saved:
        table = pd.read_csv(SAVED, index_col="FlightId")
    else:
        # Training takes hours at 30%, so the predictions are saved for reruns
        table = test_predictions(model_v3.load(args.db, args.pct, args.seed), args.db, args.weather)
        table.to_csv(SAVED)
        print(f"Saved predictions for {len(table):,} test flights to {SAVED.relative_to(ROOT)}")

    draws = bootstrap(table, args.resamples)
    scores, gaps = summarise(table, draws), compare(table, draws)
    pd.set_option("display.width", 200)
    print(scores.round(4).to_string(index=False))
    print()
    print(gaps.round(4).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    scores.round(4).to_csv(out / "error_bars.csv", index=False)
    gaps.round(4).to_csv(out / "error_bars_gaps.csv", index=False)
    print("\nSaved results/error_bars.csv and results/error_bars_gaps.csv")


if __name__ == "__main__":
    main()
