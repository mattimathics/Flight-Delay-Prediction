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

## Weather (optional)

`python src/download_weather.py --db data/Fly.db` downloads hourly airport weather reports for the 100 busiest airports from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml) into `data/weather/` (about 230 MB, one CSV per airport, plus `time_zones.json`). It uses `curl`, which comes with Windows 10+, macOS and Linux. Already-downloaded airports are skipped, so it can be rerun.
