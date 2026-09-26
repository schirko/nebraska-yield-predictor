"""Writes notebooks/alfalfa_1_data.ipynb. Kept as a script so the notebook's code lives in version control
as plain Python that diffs cleanly; rerun it after editing the cells below."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Alfalfa, part 1: the data, before any model

Alfalfa is Nebraska's main hay crop and the biggest winter feed cost for a cow-calf ranch, so a yield
model here eventually feeds Herd Planner's feed budget. Before modelling anything, this notebook asks
four plain questions of the data:

1. **How many counties report, and when?** NASS stopped publishing county hay estimates after the
   2018 crop year, so the county table runs 2000-2018 and no further.
2. **Do the counties add up to the state?** NASS still publishes a statewide yield every year. If
   the acre-weighted county average drifts away from it, the county table isn't a fair sample.
3. **How much do yields vary, and are they trending?** Alfalfa yields rise far more slowly than
   corn's, which changes how the model should handle the trend.
4. **Which weather lines up with yield?** Alfalfa is cut three to five times a season from a stand
   that lives for years, so the weather that matters is spread across the whole summer and the
   winter before, not one critical month as with corn.

Run `python scripts/fetch_alfalfa.py` first; it saves the files this notebook reads."""),
code("""from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yieldpred.alfalfa import LAST_COUNTY_YEAR, compare_with_state, coverage
from yieldpred.theme import CATEGORICAL, LIGHT

ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA = ROOT / "data" / "processed"
county = pd.read_parquet(DATA / "ne_alfalfa_county.parquet")
state = pd.read_parquet(DATA / "ne_alfalfa_state.parquet")
table = pd.read_parquet(DATA / "alfalfa_table_ne.parquet")

plt.rcParams.update({"figure.figsize": (9, 3.6), "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": LIGHT["grid"], "axes.edgecolor": LIGHT["axis"],
                     "axes.facecolor": LIGHT["surface"], "figure.facecolor": LIGHT["surface"]})
BLUE, EARTH = CATEGORICAL[0], CATEGORICAL[1]
print(f"County rows: {len(county):,}  counties: {county['fips'].nunique()}  "
      f"years: {county['year'].min()}-{county['year'].max()}")
print(f"State rows: {len(state)}  years: {state['year'].min()}-{state['year'].max()}")
print(f"Model table: {len(table):,} county-years with a published yield and weather")"""),
md("""## 1. How many counties report each year?

A county's yield can be withheld ("(D)") to protect individual farms, even when its acres are
published. So there are two counts: counties listed at all, and counties with a yield we can use."""),
code("""cov = coverage(county, state)
display(cov)

fig, ax = plt.subplots()
ax.bar(cov["year"], cov["counties_with_yield"], color=BLUE, width=0.7)
ax.set_title("Nebraska counties with a published alfalfa yield", loc="left")
ax.set_ylabel("counties")
ax.set_xlim(1999.3, LAST_COUNTY_YEAR + 0.7)
plt.show()

if "share_of_state_acres" in cov:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(cov["year"], cov["share_of_state_acres"] * 100, color=BLUE, lw=2, marker="o", ms=4)
    ax.set_title("Share of the state's alfalfa acres in the county table", loc="left")
    ax.set_ylabel("% of state acres")
    ax.set_ylim(0, 105)
    plt.show()"""),
md("""**Reading it:** a steady count means the model sees the same places each year. A falling count
means later years lean on fewer, usually bigger, counties, which would make those years look
different for reasons that have nothing to do with weather."""),
md("""## 2. Do the counties add up to the state?

The acre-weighted average of the county yields should sit close to NASS's own state yield. The
state line keeps going after 2018; the county line can't."""),
code("""check = compare_with_state(county, state)
st = state[state["practice"] == "all"].sort_values("year")

fig, ax = plt.subplots()
ax.plot(st["year"], st["yield_tons_acre"], color=EARTH, lw=2, label="NASS state yield")
ax.plot(check["year"], check["county_weighted_yield"], color=BLUE, lw=2, marker="o", ms=4,
        label="County average (acre-weighted)")
ax.axvline(LAST_COUNTY_YEAR + 0.5, color=LIGHT["axis"], lw=1)
ax.text(LAST_COUNTY_YEAR + 0.7, ax.get_ylim()[1], "county estimates end", va="top", fontsize=9,
        color=LIGHT["muted"])
ax.set_title("Alfalfa yield, tons per acre", loc="left")
ax.legend(frameon=False, loc="lower left")
plt.show()

diff = check["difference"].dropna()
print(f"County average minus state yield: mean {diff.mean():+.2f}, largest gap {diff.abs().max():.2f} tons/acre")"""),
md("""**Reading it:** gaps of a few hundredths of a ton are rounding. A gap that is consistently
positive would mean the published counties are the better ones (often the irrigated ones), so a
model trained on them would run high for the state."""),
md("""## 3. Irrigated against dryland

Where NASS split the yield by practice, irrigation's effect on alfalfa can be measured directly,
as it was for corn."""),
code("""practices = county.groupby(["year", "practice"])["yield_tons_acre"].median().unstack()
if {"irrigated", "non_irrigated"} <= set(practices.columns):
    fig, ax = plt.subplots()
    ax.plot(practices.index, practices["irrigated"], color=BLUE, lw=2, label="Irrigated")
    ax.plot(practices.index, practices["non_irrigated"], color=EARTH, lw=2, label="Dryland")
    ax.set_title("Median county alfalfa yield by practice, tons per acre", loc="left")
    ax.legend(frameon=False)
    plt.show()
    gap = (practices["irrigated"] - practices["non_irrigated"]).dropna()
    print(f"Irrigation advantage: {gap.mean():.2f} tons/acre on average; largest in {int(gap.idxmax())} ({gap.max():.2f})")
else:
    print("NASS didn't split Nebraska county alfalfa by practice in these years; only the all-practice yield exists.")"""),
md("""## 4. How much do yields vary, and is there a trend?"""),
code("""by_year = table.groupby("year")["yield_tons_acre"]
band = pd.DataFrame({"p10": by_year.quantile(0.1), "median": by_year.median(), "p90": by_year.quantile(0.9)})

fig, ax = plt.subplots()
ax.fill_between(band.index, band["p10"], band["p90"], color=BLUE, alpha=0.18, lw=0,
                label="middle 80% of counties")
ax.plot(band.index, band["median"], color=BLUE, lw=2, label="median county")
ax.set_title("Spread of county alfalfa yields, tons per acre", loc="left")
ax.legend(frameon=False, loc="lower left")
plt.show()

slope = np.polyfit(band.index, band["median"], 1)[0]
print(f"Trend in the median county: {slope:+.3f} tons/acre per year "
      f"({slope / band['median'].mean() * 100:+.1f}% a year)")
print(f"Spread between counties (p90 - p10), typical year: {(band['p90'] - band['p10']).median():.2f} tons/acre")"""),
md("""**Reading it:** corn yields in Nebraska's median county climbed about 2.3 bu/acre a year from 2000, about 1.4% a year. If alfalfa's
trend is much flatter, the model barely needs a trend term, which removes the extrapolation problem
that tripped up gradient boosting on corn. The spread between counties is the part soils, water
and management explain, and weather has to explain what's left."""),
md("""## 5. Which weather lines up with yield?

Comparing raw yields with raw weather would mostly compare the irrigated west with the dryland
east. So both are turned into **within-county anomalies** first: each county's yield minus its own
average, and the same for each weather feature. What's left is the year-to-year swing, which is
what weather can explain."""),
code("""features = ["gdd_season", "precip_season_mm", "precip_early_mm", "precip_late_mm",
            "heat_days", "winter_min_c", "freeze_days"]
anom = table[["fips", "yield_tons_acre"] + features].copy()
cols = ["yield_tons_acre"] + features
anom[cols] = anom[cols] - anom.groupby("fips")[cols].transform("mean")

corr = (anom[features].corrwith(anom["yield_tons_acre"], method="spearman")
        .rename("correlation with yield (within county)").sort_values(key=abs, ascending=False))
display(corr.round(2).to_frame())

top = corr.index[0]
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(anom[top], anom["yield_tons_acre"], s=10, color=BLUE, alpha=0.45, lw=0)
ax.axhline(0, color=LIGHT["axis"], lw=1)
ax.axvline(0, color=LIGHT["axis"], lw=1)
ax.set_xlabel(f"{top}, difference from the county's average")
ax.set_ylabel("yield difference, tons/acre")
ax.set_title(f"The strongest single link: {top}", loc="left")
plt.show()"""),
md("""**Reading it:** a correlation is a hint, not a result. Season rain and early-summer rain should
matter most on dryland; heat days should hurt regrowth; a hard winter should show up the next
year. Features that correlate with each other (a hot year is usually a dry year) will share the
credit, which is why the next notebook uses an ablation, as the corn model did.

## What this sets up

The next milestone fits the model: the same honest validation as corn (leave-district-out for
unseen counties, and training through 2014 to predict 2015-2018 for unseen years), with the
statewide series after 2018 as an out-of-sample check that the county model still adds up."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "alfalfa_1_data.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"Wrote {out}")
