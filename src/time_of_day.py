"""
Delay rate by time of day: scheduled departure time vs actual departure time.

The coursework chart grouped flights by the time they actually left. That makes
night look worst, because a delayed evening flight that finally leaves after
9 pm gets counted as a "night" flight. A passenger booking a ticket only knows
the scheduled time, so this script groups flights both ways and draws them side
by side.

Usage:
    python src/time_of_day.py --db data/Fly.db

It prints the delay rate for each time slot and saves
images/delay_rate_by_time_of_day.png. It reads the database and never changes it.
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
SLOTS = ["Morning", "Afternoon", "Evening", "Night"]

# Same boundaries as the coursework notebook, so the numbers stay comparable.
SLOT_HOURS = {
    "Morning": "05:00-11:59",
    "Afternoon": "12:00-16:59",
    "Evening": "17:00-20:59",
    "Night": "21:00-04:59",
}


def slot_case(column):
    """SQL that turns an hhmm time (e.g. 1730) into a time-of-day slot name."""
    return f"""CASE
        WHEN {column} BETWEEN 500 AND 1159 THEN 'Morning'
        WHEN {column} BETWEEN 1200 AND 1659 THEN 'Afternoon'
        WHEN {column} BETWEEN 1700 AND 2059 THEN 'Evening'
        ELSE 'Night'
    END"""


def rate_query(time_column):
    """SQL that counts flights and late flights (15+ min) per slot."""
    # Only flights that arrived have an ArrDelay; cancelled and diverted ones are left out.
    return f"""
        SELECT {slot_case(time_column)} AS Slot,
               COUNT(*) AS Flights,
               SUM(ArrDelay >= 15) AS Late
        FROM flightdetails
        WHERE ArrDelay IS NOT NULL
        GROUP BY Slot
    """


def delay_rates(db_path):
    """Return one row per slot with the delay rate grouped both ways."""
    # Open read-only so running this can never damage the database.
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        scheduled = pd.read_sql_query(rate_query("CRSDepTime"), conn)
        actual = pd.read_sql_query(rate_query("DepTime"), conn)
    table = scheduled.merge(actual, on="Slot", suffixes=("Scheduled", "Actual"))
    table["RateScheduled"] = table["LateScheduled"] / table["FlightsScheduled"]
    table["RateActual"] = table["LateActual"] / table["FlightsActual"]
    # Put the slots in the order of the day instead of alphabetical order.
    return table.set_index("Slot").loc[SLOTS].reset_index()


def draw_chart(table, out_path):
    """Grouped bar chart: scheduled time (the honest view) next to actual time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(table))
    width = 0.38
    # Scheduled gets the strong colour because it is the view a passenger can act on.
    bars_sched = ax.bar([i - width / 2 for i in x], table["RateScheduled"], width,
                        color="#2a78d6", label="By scheduled departure (what a passenger books)")
    bars_actual = ax.bar([i + width / 2 for i in x], table["RateActual"], width,
                         color="#c3c2bc", label="By actual departure (coursework chart)")
    for bars in (bars_sched, bars_actual):
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in bars.datavalues],
                     padding=3, fontsize=9, color="#2b2b2a")

    ax.set_xticks(list(x), [f"{s}\n{SLOT_HOURS[s]}" for s in table["Slot"]])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of flights arriving 15+ min late")
    ax.set_ylim(0, max(table["RateScheduled"].max(), table["RateActual"].max()) * 1.18)
    ax.set_title("Evening is the worst time to book a flight", loc="left",
                 fontsize=13, fontweight="bold", pad=40)
    ax.text(0, 1.02, "US domestic flights 2004 to April 2008. Grouping by actual departure "
            "makes night look worst,\nbecause delayed evening flights get pushed past 9 pm.",
            transform=ax.transAxes, fontsize=9, color="#52514e", va="bottom")
    # Light styling so the bars, not the frame, carry the message.
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_table(table):
    """Print the numbers the README quotes, so each one can be traced to this run."""
    print(f"{'Slot':<10} {'Scheduled':>10} {'Actual':>10} {'Flights (sched.)':>18}")
    for row in table.itertuples():
        print(f"{row.Slot:<10} {row.RateScheduled:>10.1%} {row.RateActual:>10.1%} {row.FlightsScheduled:>18,}")
    print(f"Total flights counted: {table['FlightsScheduled'].sum():,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default=str(ROOT / "data" / "Fly.db"), help="path to Fly.db")
    parser.add_argument("--out", default=str(ROOT / "images" / "delay_rate_by_time_of_day.png"))
    args = parser.parse_args()

    table = delay_rates(args.db)
    print_table(table)
    draw_chart(table, args.out)
    print(f"Saved chart to {args.out}")


if __name__ == "__main__":
    main()
