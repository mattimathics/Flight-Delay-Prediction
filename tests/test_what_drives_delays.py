"""Tests for src/what_drives_delays.py on the tiny synthetic database."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import model_v2
import what_drives_delays as wdd


def test_every_model_column_has_a_plain_name():
    _, num, cat = model_v2.SETUPS[wdd.MODEL]
    assert set(num + cat) == set(wdd.PLAIN_NAMES)


def test_a_column_the_model_ignores_has_no_importance():
    # The outcome depends only on 'signal', so shuffling 'noise' can't change the score
    rng = np.random.default_rng(0)
    flights = pd.DataFrame({"signal": rng.normal(size=500), "noise": rng.normal(size=500)})
    flights["Delayed"] = (flights["signal"] > 0).astype(int)
    # A model that is handed both columns but only reads 'signal'
    reads_signal_only = ColumnTransformer([("keep", "passthrough", ["signal"])])
    model = Pipeline([("pick", reads_signal_only), ("model", LogisticRegression())])
    model.fit(flights[["signal", "noise"]], flights["Delayed"])

    table = wdd.importance_table(model, flights, ["signal", "noise"])
    assert list(table["feature"]) == ["signal", "noise"]
    assert table.loc[1, "auc_drop"] == 0
    assert table.loc[0, "auc_drop"] > 0.3


def test_runs_end_to_end_on_the_synthetic_database(db):
    train, test = model_v2.split_by_year(model_v2.load(db, pct=100))
    model, columns = wdd.fit_model(train)
    table = wdd.importance_table(model, test, columns, n_repeats=2)
    assert len(table) == len(columns)
    assert table["auc_drop"].is_monotonic_decreasing
    ax = wdd.plot_importance(table)
    assert len(ax.patches) == len(columns)
