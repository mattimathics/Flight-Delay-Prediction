"""
Do the demo's per-flight reasons agree with Shapley values (what SHAP estimates)?

src/explain_flights.py measures each group of facts on its own: swap that group for
ordinary flights' values and see how far the chance moves. SHAP instead averages a
group's effect over every combination of the other groups, so the pushes add up
exactly to the flight's chance minus an ordinary flight's chance.

With only 9 groups, Shapley values can be computed exactly: score each flight under
all 2^9 = 512 combinations of "this flight's values" and "ordinary values", against
the same 20 ordinary training flights the demo used. The shap package estimates this
same quantity, so no extra package is needed and nothing is approximated.

Writes results/reasons_vs_shapley.csv: per group, how often each method calls it the
biggest reason, plus how often the two methods agree.

Usage (after src/explain_flights.py has saved data/model_2h.joblib):
    python src/check_reasons.py --show-pct 1 --flights 2000
"""

import argparse
from math import factorial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_web_page
import explain_flights

ROOT = Path(__file__).resolve().parents[1]
GROUPS = list(explain_flights.GROUPS)


def coalitions(n_groups):
    """Every on/off combination of the groups: row k, column g is True if bit g of k is set."""
    k = np.arange(2 ** n_groups)
    return (k[:, None] >> np.arange(n_groups)) & 1 == 1


def shapley_weights(n_groups):
    """Weight of a coalition of size s in the Shapley formula: s! (n - s - 1)! / n!."""
    return np.array([factorial(s) * factorial(n_groups - s - 1) / factorial(n_groups) for s in range(n_groups)])


def coalition_values(model, flight, references, masks):
    """Average chance of delay for each coalition: grouped facts from the flight where on, else ordinary."""
    n_masks, n_refs = len(masks), len(references)
    rows = {}
    for col in explain_flights.FEATURES:
        group = GROUPS.index(next(g for g, cols in explain_flights.GROUPS.items() if col in cols))
        own = np.repeat(masks[:, group], n_refs)  # one row per (coalition, ordinary flight)
        ordinary = np.tile(references[col].to_numpy(), n_masks)
        rows[col] = np.where(own, flight[col], ordinary)
    table = pd.DataFrame(rows).astype(references[explain_flights.FEATURES].dtypes.to_dict())
    chance = model.predict_proba(table)[:, 1]
    return chance.reshape(n_masks, n_refs).mean(axis=1) * 100  # points, 0-100


def shapley(values, masks):
    """Exact Shapley value of each group from the 2^n coalition values."""
    n = masks.shape[1]
    weights = shapley_weights(n)
    sizes = masks.sum(axis=1)
    phi = np.zeros(n)
    for g in range(n):
        without = np.flatnonzero(~masks[:, g])
        with_g = without | (1 << g)  # the same coalition with group g switched on
        phi[g] = (weights[sizes[without]] * (values[with_g] - values[without])).sum()
    return phi


def compare(own_pushes, shap_values):
    """Per group: how often each method calls it a flight's biggest reason; plus overall agreement."""
    own_top = own_pushes.abs().idxmax(axis=1)
    shap_top = shap_values.abs().idxmax(axis=1)
    table = pd.DataFrame({
        "group": GROUPS,
        "biggest_reason_demo": [(own_top == g).mean() for g in GROUPS],
        "biggest_reason_shapley": [(shap_top == g).mean() for g in GROUPS],
        "mean_points_demo": [own_pushes[g].abs().mean() for g in GROUPS],
        "mean_points_shapley": [shap_values[g].abs().mean() for g in GROUPS],
        # Same direction: both say the group raises the chance, or both say it lowers it
        "same_direction": [(np.sign(own_pushes[g]) == np.sign(shap_values[g])).mean() for g in GROUPS],
    }).sort_values("biggest_reason_shapley", ascending=False)
    # The demo shows the top three; how often does each of them appear in Shapley's top three?
    own_three = own_pushes.abs().apply(lambda r: set(r.nlargest(3).index), axis=1)
    shap_three = shap_values.abs().apply(lambda r: set(r.nlargest(3).index), axis=1)
    agreement = {
        "flights": len(own_pushes),
        "same_biggest_reason": float((own_top == shap_top).mean()),
        "top_three_overlap": float(np.mean([len(a & b) / 3 for a, b in zip(own_three, shap_three)])),
    }
    return table, agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-pct", type=float, default=None, help="the --show-pct the page was built with")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flights", type=int, default=2000, help="page flights to check (each needs 512 x 20 predictions)")
    args = ap.parse_args()
    explain_flights.check_groups()

    model, references, test = joblib.load(explain_flights.SAVED_MODEL)
    shown = build_web_page.flights_to_show(test, args.show_pct, args.seed)
    # A random draw of page flights, since exact Shapley values for all of them would take hours
    flights = shown.sample(min(args.flights, len(shown)), random_state=args.seed).reset_index(drop=True)

    own = explain_flights.group_pushes(model, flights, references)
    masks = coalitions(len(GROUPS))
    shap_rows, gaps = [], []
    for i, flight in flights.iterrows():
        values = coalition_values(model, flight, references, masks)
        phi = shapley(values, masks)
        shap_rows.append(phi)
        # Shapley values must add up to the flight's chance minus an ordinary flight's chance
        gaps.append(abs(phi.sum() - (values[-1] - values[0])))
        if (i + 1) % 200 == 0:
            print(f"{i + 1:,} of {len(flights):,} flights")
    shap_values = pd.DataFrame(shap_rows, columns=GROUPS)
    print(f"Largest gap between Shapley total and the chance difference: {max(gaps):.2e} points")

    table, agreement = compare(own, shap_values)
    print(table.round(3).to_string(index=False))
    print(agreement)
    out = ROOT / "results"
    table.round(4).to_csv(out / "reasons_vs_shapley.csv", index=False)
    pd.DataFrame([agreement]).round(4).to_csv(out / "reasons_vs_shapley_agreement.csv", index=False)
    print("\nSaved results/reasons_vs_shapley.csv and results/reasons_vs_shapley_agreement.csv")


if __name__ == "__main__":
    main()
