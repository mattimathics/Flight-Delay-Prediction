# ✈️ Flight Delay Analysis and Prediction

[![tests](https://github.com/mattimathics/Flight-Delay-Prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/mattimathics/Flight-Delay-Prediction/actions/workflows/tests.yml)

**What makes US flights late, and can we predict a delay before the plane leaves?** An end-to-end analysis of US domestic flights from 2004 to 2008, from building a SQL database to comparing machine learning models, in both **R** and **Python**.

![Delay rate by time of day](images/delay_rate_by_time_of_day.png)

## The problem

Flight delays cost airlines money and passengers time. Using the [ASA Data Expo 2009](https://doi.org/10.7910/DVN/HG7NV7) dataset of US domestic flights, this project answers five questions:

1. When is the best time of day, day of week and time of year to fly?
2. Do older planes suffer more delays?
3. How has flight volume between states changed over time?
4. Can we detect cascading delays, where one late flight makes the next one late?
5. Can we build a model that predicts whether a flight will arrive 15+ minutes late?

## Key findings

| Question | Finding |
|---|---|
| Time of day | Morning departures are the most reliable (15.5% delayed by scheduled time); risk peaks in the evening (30.6%)* |
| Day and season | Friday is the riskiest day (27.3%) and winter the riskiest season (28.3%); autumn is the most reliable (19.2%) |
| Aircraft age | Planes over 20 years old are delayed slightly more often than planes under 10 (27.0% vs 24.2%) |
| Flight volume | Flight numbers began falling nationwide in December 2007 |
| Cascading delays | Matching aircraft tail numbers traced how a late arrival at one airport delays the same plane's next departure |
| Prediction | See [Modelling](#modelling-and-a-lesson-in-target-leakage) below |

\*The coursework grouped flights by *actual* departure time, which made night look worst (40.9%). That is an artefact: a delayed evening flight slides into the "night" bucket. Grouped by *scheduled* time (v2), night drops to 26.5% and evening is the real peak. The chart above shows the original actual-time grouping.

## Modelling, and a lesson in target leakage

**Coursework version.** I compared six classifiers: logistic regression, penalised logistic regression (L1 / glmnet), decision tree, random forest, gradient boosting and a linear SVM. I used grid-search tuning in scikit-learn and a cross-validated benchmark in mlr3. Every model scored an AUC between 0.93 and 0.98.

![ROC curves, coursework version](images/roc_curve_python.png)

**Why those scores are too good.** The features included departure delay, taxi times and air time. These values are only known once the flight has left, and departure delay almost decides whether the arrival is 15+ minutes late. That is **target leakage**. A sign of it is that even plain logistic regression matched random forest.

**v2: pre-departure features only** (`src/model_v2.py`). This version rebuilds the model with only what is known at booking time: scheduled times, route, carrier, distance, month, weekday and aircraft age. It trains on 2004–2007 and tests on 2008, mimicking real forecasting, and reports the leaky model next to it so the gap is visible.

| Model | Features | Test AUC (Jan–Apr 2008) |
|---|---|---|
| Logistic regression | Pre-departure | 0.627 |
| Gradient boosting | Pre-departure | **0.641** |
| Gradient boosting | + departure delay, taxi, air time (leaky) | 0.997 |

Trained on a random 1% sample (282,428 flights from 2004–2007) and tested on 23,131 flights from 2008. The 2008 file only covers January to April.

![ROC curves, v2: pre-departure vs leaky features](images/roc_curve_v2.png)

**What this means.** Predicting a delay from the schedule alone is genuinely hard: an AUC of about 0.64 is a modest but real signal. Adding just the departure delay jumps it to 0.997. That gap is the leakage, and it is why the coursework scores were too good to be true.

| Time of day | Delay rate, by actual departure | Delay rate, by scheduled departure |
|---|---|---|
| Morning | 13.9% | 15.5% |
| Afternoon | 23.1% | 24.4% |
| Evening | 30.9% | **30.6%** |
| Night | 40.9% | 26.5% |

## Tech stack

- **Data:** SQLite (via `sqlite3` and `DBI`/`RSQLite`), with indexes on tail number, date and route
- **Python:** pandas, NumPy, statsmodels, scikit-learn, Matplotlib, seaborn, Plotly
- **R:** dplyr, ggplot2, mlr3 (+ learners, tuning, pipelines), glmnet, ranger, XGBoost

## Design decisions

- **SQLite instead of in-memory data frames.** Five years of flights (31 million rows) don't fit comfortably in laptop memory. Loading the CSVs into an indexed SQLite database let each query and each model pull just the rows it needed.
- **Delay *rates*, not counts.** Busy time slots naturally have more delays, so every comparison uses the share of flights delayed.
- **FAA definition of a delay:** arrival 15+ minutes after schedule.
- **Time-based test split in v2.** A random split lets the model peek at the future; training on earlier years and testing on 2008 does not.
- **Repeatable sampling.** SQLite's `RANDOM()` can't be seeded, so v2 keeps a flight when a fixed hash of its row id falls under the cutoff. The same database, `--pct` and `--seed` always give the same flights, so every reported number can be rerun exactly.

## Project structure

```
├── notebooks/flight_delay_analysis.ipynb   # Python analysis (Q1–Q5), outputs included
├── r/flight_delay_analysis.Rmd             # Same analysis in R with mlr3 benchmark
├── src/model_v2.py                         # Leakage-free model, time-based split
├── tests/                                  # pytest checks on a synthetic database
├── data/README.md                          # How to download the data
├── images/                                 # Charts used in this README
└── requirements.txt                        # pinned versions (+ requirements-dev.txt for tests)
```

## How to run

1. Download the data into `data/` (see [data/README.md](data/README.md)).
2. Python:
   ```bash
   pip install -r requirements.txt
   jupyter notebook notebooks/flight_delay_analysis.ipynb   # builds data/Fly.db
   python src/model_v2.py --db data/Fly.db          # same --pct and --seed give the same sample
   ```
3. R: open `r/flight_delay_analysis.Rmd` in RStudio and knit.
4. Tests (no data needed, they build a tiny synthetic database):
   ```bash
   pip install -r requirements-dev.txt
   pytest
   ```

## What I learned

- **Check what a feature knows, and when.** An AUC near 1 on a noisy real-world problem is a warning sign, not a win.
- **Plan indexes around the queries.** The cascading-delay analysis joins 31 million rows to themselves on tail number and date, so those are the columns I indexed.
- **How you bucket a variable can bias a result**, as with actual vs scheduled departure time.
- **Doing the same analysis in R and Python** showed me where mlr3 and scikit-learn differ in pipelines, tuning and resampling.

## Next steps

- Add weather data for origin airports
- Probability calibration and a cost-based decision threshold
- Package the v2 model behind a small prediction API

## About

Built by **Matthew James Tan** as part of the BSc Data Science and Business Analytics (University of London, SIM), and extended afterwards. [LinkedIn](https://www.linkedin.com/in/matthew-james-tan-461b98398) · [GitHub](https://github.com/mattimathics)

## License

MIT. Data: ASA Data Expo 2009, Harvard Dataverse.
