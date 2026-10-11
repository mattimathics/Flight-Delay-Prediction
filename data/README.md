# Data

The raw data is not stored in this repo; it's several GB.

Download from the **ASA Data Expo 2009** on Harvard Dataverse: https://doi.org/10.7910/DVN/HG7NV7

Put these files in this `data/` folder:

| File | Contents |
|---|---|
| `2004.csv.bz2` … `2008.csv.bz2` | One row per US domestic flight |
| `airports.csv` | Airport codes, city, state, coordinates |
| `carriers.csv` | Airline codes and names |
| `plane-data.csv` | Aircraft by tail number, incl. manufacture year |

Running the notebook builds `Fly.db` (SQLite) here. It is git-ignored.
