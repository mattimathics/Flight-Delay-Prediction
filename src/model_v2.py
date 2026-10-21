"""
Flight delay prediction, v2: only information known before departure.

The coursework model used DepDelay, DepTime, AirTime, TaxiIn and TaxiOut as
features. These are only known once the flight has left (or landed), and
DepDelay alone almost determines whether the arrival is 15+ minutes late.
That is target leakage, and it is why every model scored AUC 0.93-0.98.

This script rebuilds the model with schedule-time features only, uses a
time-based split (train on 2004-2007, test on 2008), and compares it with
the leaky setup so the difference is visible.

Usage:
    python src/model_v2.py --db data/Fly.db [--pct 1] [--seed 42]

The sample is deterministic: the same database, --pct and --seed always give
the same flights, so every number this script prints can be reproduced.

The database is the SQLite file built by the notebook (tables
`flightdetails` and `planes`).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

PRE_DEPARTURE_NUM = ["CRSDepHour", "CRSArrHour", "CRSElapsedTime", "Distance", "PlaneAge"]
PRE_DEPARTURE_CAT = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
LEAKY_NUM = ["DepDelay", "TaxiOut", "TaxiIn", "AirTime"]

QUERY = """
SELECT f.Year, f.Month, f.DayOfWeek, f.CRSDepTime, f.CRSArrTime, f.CRSElapsedTime,
       f.Distance, f.UniqueCarrier, f.Origin, f.Dest,
       f.DepTime, f.DepDelay, f.TaxiOut, f.TaxiIn, f.AirTime, f.ArrDelay,
       f.Year - CAST(p.year AS INTEGER) AS PlaneAge
FROM flightdetails f
LEFT JOIN planes p ON f.TailNum = p.tailnum
WHERE f.ArrDelay IS NOT NULL
  AND in_sample(f.rowid) = 1
"""

MASK64 = (1 << 64) - 1


def row_hash(rowid, seed):
    """Deterministic 64-bit hash of a row id (splitmix64), spread evenly over 0..2^64-1."""
    z = (rowid + seed * 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def in_sample(rowid, pct, seed):
    """Keep a row if its hash falls in the lowest pct percent. Same inputs, same answer."""
    return int(row_hash(rowid, seed) % 10_000 < pct * 100)


def load(db_path, pct, seed=42):
    # SQLite's RANDOM() cannot be seeded, so sampling is done with a fixed hash of
    # each row id instead: unlike RANDOM(), rerunning gives the same sample.
    with sqlite3.connect(db_path) as conn:
        conn.create_function("in_sample", 1, lambda rowid: in_sample(rowid, pct, seed), deterministic=True)
        df = pd.read_sql_query(QUERY, conn)
    for col in ["CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "DepTime",
                "DepDelay", "TaxiOut", "TaxiIn", "AirTime", "ArrDelay", "PlaneAge", "Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ArrDelay", "CRSDepTime"])
    # Planes table has missing/zero manufacture years; treat implausible ages as unknown
    df.loc[(df["PlaneAge"] < 0) | (df["PlaneAge"] > 60), "PlaneAge"] = np.nan
    df["CRSDepHour"] = (df["CRSDepTime"] // 100).clip(0, 23)
    df["CRSArrHour"] = (df["CRSArrTime"] // 100).clip(0, 23)
    df["Delayed"] = (df["ArrDelay"] >= 15).astype(int)
    for col in PRE_DEPARTURE_CAT:
        df[col] = df[col].astype(str)
    return df


def time_of_day(hhmm):
    return pd.cut(hhmm, bins=[-1, 459, 1159, 1659, 2059, 2400],
                  labels=["Night", "Morning", "Afternoon", "Evening", "Night2"]).astype(str).replace("Night2", "Night")


def logistic(num, cat):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200), cat),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])


def boosting(num, cat):
    prep = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, max_categories=255,
                               encoded_missing_value=-1), cat),
    ])
    cat_mask = [False] * len(num) + [True] * len(cat)
    model = HistGradientBoostingClassifier(categorical_features=cat_mask, max_iter=300,
                                           learning_rate=0.1, early_stopping=True, random_state=0)
    return Pipeline([("prep", prep), ("model", model)])


SETUPS = {
    "Logistic regression (pre-departure)": (logistic, PRE_DEPARTURE_NUM, PRE_DEPARTURE_CAT),
    "Gradient boosting (pre-departure)": (boosting, PRE_DEPARTURE_NUM, PRE_DEPARTURE_CAT),
    "Gradient boosting (leaky: + DepDelay, taxi, air time)": (boosting, PRE_DEPARTURE_NUM + LEAKY_NUM, PRE_DEPARTURE_CAT),
}


def delay_rate_by_time_of_day(df):
    # Grouping by actual DepTime pushes late-running evening flights into "Night",
    # inflating its delay rate; grouping by scheduled CRSDepTime does not.
    return pd.DataFrame({
        "by_actual_departure": df.groupby(time_of_day(df["DepTime"]))["Delayed"].mean(),
        "by_scheduled_departure": df.groupby(time_of_day(df["CRSDepTime"]))["Delayed"].mean(),
    }).reindex(["Morning", "Afternoon", "Evening", "Night"])


def split_by_year(df):
    """Train on 2004-2007, test on 2008, as a real forecast would."""
    train = df[df["Year"] <= 2007]
    test = df[df["Year"] == 2008]
    if test.empty:  # fallback if 2008 is missing from the database
        test = df.sample(frac=0.2, random_state=1)
        train = df.drop(test.index)
    return train, test


def fit_and_score(train, test):
    """Fit every setup in SETUPS; return the AUC table and the fitted models."""
    results, fitted = [], {}
    for name, (build, num, cat) in SETUPS.items():
        model = build(num, cat).fit(train[num + cat], train["Delayed"])
        auc = roc_auc_score(test["Delayed"], model.predict_proba(test[num + cat])[:, 1])
        results.append({"model": name, "test_auc": round(auc, 3), "n_train": len(train), "n_test": len(test)})
        fitted[name] = (model, num + cat)
    return pd.DataFrame(results), fitted


def plot_roc(fitted, test, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    for name, (model, cols) in fitted.items():
        RocCurveDisplay.from_estimator(model, test[cols], test["Delayed"], ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Pre-departure features vs leaky features (test set: 2008)")
    ax.figure.tight_layout()
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "Fly.db"))
    ap.add_argument("--pct", type=float, default=1, help="percent of flights to sample (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42); same seed, same sample")
    args = ap.parse_args()

    df = load(args.db, args.pct, args.seed)
    print(f"Sampled {len(df):,} flights (pct={args.pct}, seed={args.seed}); "
          f"{df['Delayed'].mean():.1%} arrived 15+ min late")

    rates = delay_rate_by_time_of_day(df)
    print("\nDelay rate by time of day\n", rates.round(3))

    train, test = split_by_year(df)
    print(f"\nTrain {len(train):,} rows | Test {len(test):,} rows")

    results, fitted = fit_and_score(train, test)
    results["pct"], results["seed"] = args.pct, args.seed
    for row in results.itertuples():
        print(f"{row.model}: AUC {row.test_auc:.3f}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "model_comparison_v2.csv", index=False)
    rates.to_csv(out / "delay_rate_by_time_of_day_v2.csv")

    ax = plot_roc(fitted, test)
    ax.figure.savefig(ROOT / "images" / "roc_curve_v2.png", dpi=150)
    print(f"\nSaved results to {out} and images/roc_curve_v2.png")


if __name__ == "__main__":
    main()
