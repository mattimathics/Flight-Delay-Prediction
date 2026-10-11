# ✈️ Flight Delay Analysis and Prediction

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
| Time of day | Morning departures have the lowest delay rate (14.8%); delay risk climbs through the day* |
| Day and season | Friday is the riskiest day (27.3%) and winter the riskiest season (28.3%); autumn is the most reliable (19.2%) |
| Aircraft age | Planes over 20 years old are delayed slightly more often than planes under 10 (27.0% vs 24.2%) |
| Flight volume | Flight numbers began falling nationwide in December 2007 |
| Cascading delays | Matching aircraft tail numbers traced how a late arrival at one airport delays the same plane's next departure |
| Prediction | See [Modelling](#modelling-and-a-lesson-in-target-leakage) below |

\*Measured by actual departure time. A delayed evening flight can slide into the "night" bucket, which inflates the night rate. `src/model_v2.py` recomputes it by scheduled time.

## Modelling, and a lesson in target leakage

**Coursework version.** I compared six classifiers: logistic regression, penalised logistic regression (L1 / glmnet), decision tree, random forest, gradient boosting and a linear SVM. I used grid-search tuning in scikit-learn and a cross-validated benchmark in mlr3. Every model scored an AUC between 0.93 and 0.98.

![ROC curves, coursework version](images/roc_curve_python.png)

**Why those scores are too good.** The features included departure delay, taxi times and air time. These values are only known once the flight has left, and departure delay almost decides whether the arrival is 15+ minutes late. That is **target leakage**. A sign of it is that even plain logistic regression matched random forest.

**v2: pre-departure features only** (`src/model_v2.py`). This version rebuilds the model with only what is known at booking time: scheduled times, route, carrier, distance, month, weekday and aircraft age. It trains on 2004–2007 and tests on 2008, mimicking real forecasting, and reports the leaky model next to it so the gap is visible.

| Model | Features | Test AUC (2008) |
|---|---|---|
| Logistic regression | Pre-departure | [run v2] |
| Gradient boosting | Pre-departure | [run v2] |
| Gradient boosting | + departure delay, taxi, air time (leaky) | [run v2] |

## Tech stack

- **Data:** SQLite (via `sqlite3` and `DBI`/`RSQLite`), with indexes on tail number, date and route
- **Python:** pandas, NumPy, statsmodels, scikit-learn, Matplotlib, seaborn, Plotly
- **R:** dplyr, ggplot2, mlr3 (+ learners, tuning, pipelines), glmnet, ranger, XGBoost

## Design decisions

- **SQLite instead of in-memory data frames.** Five years of flights don't fit comfortably in laptop memory. Loading the CSVs into an indexed SQLite database cut query times from minutes to seconds, and let each model pull just the sample it needed.
- **Delay *rates*, not counts.** Busy time slots naturally have more delays, so every comparison uses the share of flights delayed.
- **FAA definition of a delay:** arrival 15+ minutes after schedule.
- **Time-based test split in v2.** A random split lets the model peek at the future; training on earlier years and testing on 2008 does not.

## Project structure

```
├── notebooks/flight_delay_analysis.ipynb   # Python analysis (Q1–Q5), outputs included
├── r/flight_delay_analysis.Rmd             # Same analysis in R with mlr3 benchmark
├── src/model_v2.py                         # Leakage-free model, time-based split
├── data/README.md                          # How to download the data
├── images/                                 # Charts used in this README
└── requirements.txt
```

## How to run

1. Download the data into `data/` (see [data/README.md](data/README.md)).
2. Python:
   ```bash
   pip install -r requirements.txt
   jupyter notebook notebooks/flight_delay_analysis.ipynb   # builds data/Fly.db
   python src/model_v2.py --db data/Fly.db
   ```
3. R: open `r/flight_delay_analysis.Rmd` in RStudio and knit.

## What I learned

- **Check what a feature knows, and when.** An AUC near 1 on a noisy real-world problem is a warning sign, not a win.
- **Indexing matters at scale.** The same SQL went from minutes to seconds once the right columns were indexed.
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
