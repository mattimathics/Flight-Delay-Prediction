"""
Delay rate by day of week, season and aircraft age, counted over every flight.

The README's day, season and aircraft-age numbers used to come from the
coursework report, not from code in this repo. This script recomputes them
from the database so each one can be traced to a run, and redraws the three
charts in the same style as the time-of-day chart.

It also fixes an aircraft-age bug in the notebook: a few planes have a build
year of 0 in plane-data.csv, which gave them an age of about 2,004 years and
put them in the "old" group. Ages below 0 or above 60 are now treated as
unknown, the same rule src/model_v2.py uses.

Usage:
    python src/findings.py --db data/Fly.db

It prints each table, saves them to results/findings_*.csv and saves three
charts to images/. It reads the database and never changes it.
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREY = "#2a78d6", "#c3c2bc"

DAY_SQL = """CASE f.DayOfWeek
    WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday' WHEN 3 THEN 'Wednesday'
    WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday' WHEN 6 THEN 'Saturday'
    ELSE 'Sunday' END"""
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Same month groups as the coursework notebook (December counts as winter).
SEASON_SQL = """CASE
    WHEN f.Month BETWEEN 3 AND 5 THEN 'Spring'
    WHEN f.Month BETWEEN 6 AND 8 THEN 'Summer'
    WHEN f.Month BETWEEN 9 AND 11 THEN 'Autumn'
    ELSE 'Winter' END"""
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]

# Age = flight year minus build year. Ages outside 0-60 are data errors (e.g. build year 0).
AGE_SQL = """CASE
    WHEN f.Year - p.year IS NULL OR f.Year - p.year < 0 OR f.Year - p.year > 60 THEN 'Unknown'
    WHEN f.Year - p.year <= 10 THEN '0-10 years'
    WHEN f.Year - p.year <= 20 THEN '11-20 years'
    ELSE 'Over 20 years' END"""
AGES = ["0-10 years", "11-20 years", "Over 20 years", "Unknown"]


def rate_query(group_sql):
    """SQL that counts flights and late flights (15+ min) per group."""
    # Only flights that arrived have an ArrDelay; cancelled and diverted ones are left out.
    # The LEFT JOIN keeps flights whose plane is missing from the planes table.
    return f"""
        SELECT {group_sql} AS "Group",
               COUNT(*) AS Flights,
               SUM(f.ArrDelay >= 15) AS Late
        FROM flightdetails f
        LEFT JOIN planes p ON f.TailNum = p.tailnum
        WHERE f.ArrDelay IS NOT NULL
        GROUP BY "Group"
    """


def delay_rate(conn, group_sql, order):
    """Return one row per group with its flight count and delay rate, in a fixed order."""
    table = pd.read_sql_query(rate_query(group_sql), conn)
    table["Rate"] = table["Late"] / table["Flights"]
    # Put groups in their natural order (Monday first) instead of alphabetical order.
    return table.set_index("Group").loc[order].reset_index()


def all_findings(db_path):
    """Run the three queries and return them by name."""
    # Open read-only so running this can never damage the database.
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return {
            "day_of_week": delay_rate(conn, DAY_SQL, DAYS),
            "season": delay_rate(conn, SEASON_SQL, SEASONS),
            "aircraft_age": delay_rate(conn, AGE_SQL, AGES),
        }


def draw_bar_chart(table, title, subtitle, out_path):
    """One bar per group; the worst group in blue so the eye goes to it first."""
    fig, ax = plt.subplots(figsize=(8, 5))
    worst = table["Rate"].idxmax()
    colours = [BLUE if i == worst else GREY for i in table.index]
    bars = ax.bar(table["Group"], table["Rate"], color=colours, width=0.6)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in table["Rate"]],
                 padding=3, fontsize=9, color="#2b2b2a")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of flights arriving 15+ min late")
    ax.set_ylim(0, table["Rate"].max() * 1.18)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=40)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#52514e", va="bottom")
    # Light styling so the bars, not the frame, carry the message.
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_all_charts(findings, image_dir):
    """Redraw the three README charts from the computed tables."""
    source = "US domestic flights 2004 to April 2008, every flight that arrived."
    draw_bar_chart(findings["day_of_week"], "Friday has the most late flights",
                   source, image_dir / "delay_rate_by_day_of_week.png")
    draw_bar_chart(findings["season"], "Winter and summer have the most late flights",
                   source + "\nWinter = Dec, Jan, Feb. 2008 only covers January to April.",
                   image_dir / "delay_rate_by_season.png")
    # Leave "Unknown" off the age chart: it is missing data, not an age group.
    known_ages = findings["aircraft_age"].query("Group != 'Unknown'").reset_index(drop=True)
    draw_bar_chart(known_ages, "Plane age makes little difference to delays",
                   source + "\nAge = flight year minus build year. Planes with no valid build year are left out.",
                   image_dir / "delay_rate_by_aircraft_age.png")


def save_tables(findings, results_dir):
    """Save each table as a CSV so every README number has a file behind it."""
    results_dir.mkdir(exist_ok=True)
    for name, table in findings.items():
        table.to_csv(results_dir / f"findings_{name}.csv", index=False)


def print_tables(findings):
    """Print the numbers the README quotes, so each one can be traced to this run."""
    for name, table in findings.items():
        print(f"\n{name}")
        for row in table.itertuples():
            print(f"  {row.Group:<14} {row.Rate:>6.1%}  of {row.Flights:>12,} flights")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default=str(ROOT / "data" / "Fly.db"), help="path to Fly.db")
    args = parser.parse_args()

    findings = all_findings(args.db)
    print_tables(findings)
    save_tables(findings, ROOT / "results")
    draw_all_charts(findings, ROOT / "images")
    print("\nSaved tables to results/ and charts to images/")


if __name__ == "__main__":
    main()
