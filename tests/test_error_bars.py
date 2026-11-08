"""Tests for src/error_bars.py."""

import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import error_bars
import model_v3


def random_flights(n=500, seed=0):
    rng = np.random.default_rng(seed)
    late = rng.integers(0, 2, n)
    # Rounded so many flights share a risk, to test that ties count as half right
    risk = np.round(np.clip(0.3 * late + rng.random(n) * 0.7, 0, 1), 2)
    return late, risk


def test_auc_matches_sklearn_with_ties():
    late, risk = random_flights()
    scorer = error_bars.Scorer(late, risk)
    assert np.isclose(scorer.auc(np.ones(len(late), dtype=int)), roc_auc_score(late, risk))


def test_resampled_auc_matches_scoring_the_resampled_flights():
    late, risk = random_flights()
    picks = np.random.default_rng(1).integers(0, len(late), len(late))
    counts = np.bincount(picks, minlength=len(late))
    # Counting flight i counts[i] times must equal scoring the resampled list itself
    assert np.isclose(error_bars.Scorer(late, risk).auc(counts), roc_auc_score(late[picks], risk[picks]))


def test_warning_list_matches_model_v3_without_ties():
    rng = np.random.default_rng(2)
    late, risk = rng.integers(0, 2, 1000), rng.random(1000)
    got = error_bars.Scorer(late, risk).late_rate_in_riskiest(np.ones(1000, dtype=int))
    assert np.isclose(got, model_v3.flag_riskiest(late, risk)["late_rate_in_flagged"])


def test_end_to_end_on_the_small_database(db):
    # The random database never lands a plane where it next departs from; make some do
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET Dest = 'ORD' WHERE rowid % 3 = 0")
    table = error_bars.test_predictions(model_v3.load(db, pct=100, seed=42), db)
    draws = error_bars.bootstrap(table, n_resamples=20)
    scores = error_bars.summarise(table, draws)
    # No weather folder, so no weather model
    assert list(scores["model"]) == [name for col, name in error_bars.MODELS.items() if col in table]
    assert len(scores) == 5
    assert (scores["auc_low"] <= scores["auc_high"]).all()
    gaps = error_bars.compare(table, draws)
    assert len(gaps) == 4
    assert gaps["share_of_resamples_better"].between(0, 1).all()
    assert isinstance(table, pd.DataFrame) and table["late"].isin([0, 1]).all()
