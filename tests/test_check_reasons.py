"""Tests for src/check_reasons.py."""

import numpy as np
import pandas as pd

import check_reasons
import explain_flights


class AdditiveModel:
    """Chance = 0.2 + previous delay / 200 + 0.1 if it leaves at 18:00 or later: no interactions."""

    def predict_proba(self, X):
        p = 0.2 + X["PlanePrevDepDelay"].astype(float).fillna(0) / 200 + 0.1 * (X["CRSDepHour"].astype(float) >= 18)
        p = p.clip(0, 1).to_numpy()
        return np.c_[1 - p, p]


class BothNeededModel:
    """Chance is high only when the plane is late AND it leaves in the evening: a pure interaction."""

    def predict_proba(self, X):
        both = (X["PlanePrevDepDelay"].astype(float).fillna(0) > 30) & (X["CRSDepHour"].astype(float) >= 18)
        p = np.where(both, 0.9, 0.1)
        return np.c_[1 - p, p]


def flight_and_references():
    references = pd.DataFrame({col: [1.0, 1.0] for col in explain_flights.FEATURES})
    for col in ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        references[col] = ["1", "2"]
    references["PlanePrevDepDelay"] = [0.0, 0.0]
    references["CRSDepHour"] = [8.0, 9.0]
    flight = references.iloc[0].copy()
    flight["PlanePrevDepDelay"], flight["CRSDepHour"] = 60.0, 19.0
    return flight, references


def test_shapley_values_add_up_and_match_the_demo_when_there_are_no_interactions():
    flight, references = flight_and_references()
    masks = check_reasons.coalitions(len(check_reasons.GROUPS))
    values = check_reasons.coalition_values(AdditiveModel(), flight, references, masks)
    phi = dict(zip(check_reasons.GROUPS, check_reasons.shapley(values, masks)))
    assert np.isclose(sum(phi.values()), values[-1] - values[0])
    # Without interactions, swapping one group alone gives the same answer as Shapley
    own = explain_flights.group_pushes(AdditiveModel(), pd.DataFrame([flight]), references).iloc[0]
    assert np.isclose(phi["plane"], 30) and np.isclose(own["plane"], 30)
    assert np.isclose(phi["time"], 10) and np.isclose(own["time"], 10)


def test_with_an_interaction_shapley_splits_the_credit_but_one_at_a_time_counts_it_twice():
    flight, references = flight_and_references()
    masks = check_reasons.coalitions(len(check_reasons.GROUPS))
    values = check_reasons.coalition_values(BothNeededModel(), flight, references, masks)
    phi = dict(zip(check_reasons.GROUPS, check_reasons.shapley(values, masks)))
    own = explain_flights.group_pushes(BothNeededModel(), pd.DataFrame([flight]), references).iloc[0]
    # The 80-point rise needs both facts: Shapley gives each half, swapping one alone gives each all of it
    assert np.isclose(phi["plane"], 40) and np.isclose(phi["time"], 40)
    assert np.isclose(own["plane"], 80) and np.isclose(own["time"], 80)


def test_compare_counts_agreement():
    groups = check_reasons.GROUPS
    own = pd.DataFrame([[5 if g == "plane" else 0 for g in groups], [0 if g != "time" else -3 for g in groups]],
                       columns=groups)
    shap_values = own.copy()
    table, agreement = check_reasons.compare(own, shap_values)
    assert agreement["same_biggest_reason"] == 1.0
    assert np.isclose(table["biggest_reason_demo"].sum(), 1)


def test_runs_on_a_real_model_from_the_small_database(db):
    import sqlite3

    import model_v3
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE flightdetails SET Dest = 'ORD' WHERE rowid % 3 = 0")
    model, train, test = explain_flights.train_2h_model(model_v3.load(db, pct=100, seed=42), db)
    references = train.sample(20, random_state=0)
    masks = check_reasons.coalitions(len(check_reasons.GROUPS))
    flight = test.iloc[0]
    values = check_reasons.coalition_values(model, flight, references, masks)
    phi = check_reasons.shapley(values, masks)
    assert np.isclose(phi.sum(), values[-1] - values[0])
    # All facts from the flight itself: the same as scoring the flight directly
    assert np.isclose(values[-1], model.predict_proba(test[explain_flights.FEATURES].iloc[[0]])[0, 1] * 100)
