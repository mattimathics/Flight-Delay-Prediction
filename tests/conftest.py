import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def db(tmp_path):
    """A small flightdetails/planes database shaped like the real Fly.db."""
    rng = np.random.default_rng(0)
    n = 2000
    flights = pd.DataFrame({
        "Year": rng.choice([2006, 2007, 2008], n),
        "Month": rng.integers(1, 13, n),
        "DayofMonth": rng.integers(1, 29, n),
        "DayOfWeek": rng.integers(1, 8, n),
        "CRSDepTime": rng.integers(0, 2400, n),
        "CRSArrTime": rng.integers(0, 2400, n),
        "CRSElapsedTime": rng.integers(40, 300, n),
        "Distance": rng.integers(100, 2500, n),
        "UniqueCarrier": rng.choice(["AA", "UA", "WN"], n),
        "Origin": rng.choice(["ORD", "ATL", "SFO"], n),
        "Dest": rng.choice(["LAX", "JFK", "DEN"], n),
        "DepTime": rng.integers(0, 2400, n),
        "DepDelay": rng.integers(-10, 60, n),
        "TaxiOut": rng.integers(5, 30, n),
        "TaxiIn": rng.integers(2, 15, n),
        "AirTime": rng.integers(30, 280, n),
        "ArrTime": rng.integers(0, 2400, n),
        "ActualElapsedTime": rng.integers(40, 300, n),
        "ArrDelay": rng.integers(-20, 90, n).astype(float),
        "TailNum": rng.choice(["N1", "N2", "N3", "N4"], n),
    })
    flights.loc[:9, "ArrDelay"] = np.nan  # cancelled flights have no arrival delay
    planes = pd.DataFrame({
        "tailnum": ["N1", "N2", "N3", "N4"],
        "year": ["1990", "2005", "None", "0000"],  # the real file has 'None' and '0000' years
    })
    path = tmp_path / "Fly.db"
    with sqlite3.connect(path) as conn:
        flights.to_sql("flightdetails", conn, index=False)
        planes.to_sql("planes", conn, index=False)
    return path
