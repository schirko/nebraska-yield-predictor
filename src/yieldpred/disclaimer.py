"""The legal notices, in one place so five pages can't drift apart.

Deliberately free of any Streamlit import, like `appdata`: this module exports
strings, and the pages decide how to render them. That keeps the text testable
and means the README, the app footer and the limitations panel are provably the
same words rather than five copies that diverge over a year.

Three separate jobs, and they are not interchangeable:

1. **Not advice.** The model predicts county averages from public data. Someone
   acting on it to market grain, value ground or file an insurance claim is
   outside what it can support.
2. **No endorsement.** Federal data APIs require this. USDA's terms oblige a
   service to state that it uses the API but is "not endorsed or certified by"
   the agency, and forbid using the agency name to imply endorsement. The same
   pattern applies to the other federal sources here.
3. **Accuracy, in numbers.** This is the part worth caring about. A specific,
   checkable statement of typical error does more real work than any amount of
   "use at your own risk" - it is both better disclosure and better science, and
   it is only possible because the project measured its limits honestly.

If this ever ships under a company rather than as a portfolio project, these are
a starting point for a lawyer, not a substitute for one.
"""

from __future__ import annotations

# One line, small, on every page.
FOOTER = (
    "Educational project — not agronomic, financial or insurance advice. "
    "Predictions are county-level estimates with a typical error of roughly "
    "14 bu/acre. Not endorsed or certified by USDA, NASA, USGS or the Census Bureau."
)

# The federal sources, each with the no-endorsement wording their terms expect.
DATA_SOURCES = """
This product uses the **USDA NASS Quick Stats API** but is not endorsed or certified
by USDA NASS. It uses **USDA NRCS Soil Data Access**, **NASA POWER**, the **USGS
Elevation Point Query Service** and **US Census Bureau** boundary files on the same
basis: these agencies supply the data and have no involvement in, and no
responsibility for, what this project does with it.

All source data is public. Any error in the analysis is the author's.
"""

# The substance. Numbers come from the validation tables in notes/METHODS.md and
# should be updated together with them.
LIMITATIONS = """
### What This Model Cannot Do

**It is not advice.** These are county-average estimates built from public data for
learning and demonstration. Do not use them to market grain, price land, plan
insurance, or make any decision with money attached.

**Typical error is large in farming terms.** On counties the model has not seen, a
typical miss is about **14 bu/acre in Nebraska** and **10 bu/acre in Iowa** (mean
absolute error; RMSE is 18.5 and 13.0). At roughly $4.50 a bushel that is $45–65 an
acre — far too coarse for a commercial decision.

**The errors are not random.** They cluster geographically in **all 26 years** in both
states (Moran's I +0.60 Nebraska, +0.56 Iowa). Some places are wrong in the same
direction year after year, which means a specific county's error is likely to be
worse, or better, than the average above.

**Iowa's unseen-county score is optimistic by about 0.023.** Two counties served by the
same NASA POWER weather cell have identical weather rows, and 44% of Iowa's modelled counties
share theirs with a county in a different agricultural district - so the leave-district-out
test is not quite as clean as it looks there. Measured rather than assumed: 0.781 as reported,
about 0.758 with the leaking rows removed from training. Nebraska is unaffected (2%).

**It is retrospective, not a forecast.** The model explains seasons that have already
happened. It has never been built or validated to predict a crop in progress.

**Two documented data limits.** USDA stopped publishing county corn silage for
Nebraska after 2007, so the abandonment figures there cannot be interpreted for
2008 onward. And Iowa's spring-weather relationships measurably changed after 2018 —
what held through 2018 stopped holding, for reasons this project has not explained.

**Coverage shrinks over time.** USDA reported 91 Nebraska counties in 2000 and 46 in
2025, and the counties that stop reporting are not a random sample — they are the
ones growing little corn. Results describe corn-growing counties, not whole states.

The project documents its own accuracy in detail rather than claiming more than it
can support. That is the point of it.
"""

# For the README, where a reviewer reads before running anything.
README_BLOCK = """## Disclaimer

Educational and portfolio project. **Not agronomic, financial or insurance advice.**

Predictions are county averages with a typical error of roughly 14 bu/acre (Nebraska)
and 10 bu/acre (Iowa) on unseen counties, and the errors cluster geographically in
every year of the record. The model is retrospective and has never been validated for
in-season use. See the Limitations panel in the app, or `notes/METHODS.md`, for the
full accuracy picture.

This product uses the USDA NASS Quick Stats API but is not endorsed or certified by
USDA NASS. It likewise uses USDA NRCS Soil Data Access, NASA POWER, USGS and US Census
Bureau data without endorsement by those agencies. All source data is public; any error
in the analysis is mine.
"""
