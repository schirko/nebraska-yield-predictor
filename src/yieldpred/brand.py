"""The app's logo and the farm app suite's shared look.

The logo is an ear of corn on a soil-brown circle, in the same family as Herd
Planner's cow and Farm Equipment Planner's tractor: a colored circle, a cream
drawing, a wheat-gold accent. The master SVG is app/assets/logo.svg; logo.png is
rendered from it because Streamlit's tab icon and st.logo are most reliable with
a PNG.

Every page calls show_logo() right after st.set_page_config(), and passes
PAGE_ICON as its page_icon, so every page gets the logo in the browser tab and
top-left corner, the suite's header bar and button colors, and the "Our farm
apps" list at the foot of the sidebar.
"""

import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "app" / "assets"
LOGO = ASSETS / "logo.png"
PAGE_ICON = str(LOGO)
APPS_FILE = ASSETS / "suite-apps.json"   # copy of herd-planner/brand/suite-apps.json
APP_ID = "corn-yield-predictor"          # this app's id in that list

# The pieces of the suite look (app/assets/suite.css) that .streamlit/config.toml
# can't set. Streamlit's own element names (data-testid) can change between
# versions; if the header turns white again after an upgrade, check them here.
SUITE_CSS = """
<style>
header[data-testid="stHeader"] { background: #243b2f; }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a,
header[data-testid="stHeader"] [data-testid="stMainMenu"] * { color: #ffffff; }
/* Gold buttons take deep-green text: white on gold is too faint to read (2.2:1). */
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  color: #243b2f !important; font-weight: 600;
}
/* The slider's value label would be gold on paper (too faint): deep green instead. */
[data-testid="stSliderThumbValue"] { color: #243b2f; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: #e6b75a; border-color: #e6b75a; }
</style>
"""


def show_logo():
    """Put the logo in the top-left corner and apply the suite's header and button colors."""
    import streamlit as st  # imported here so the pipeline can import yieldpred without Streamlit

    st.logo(str(LOGO), size="large", icon_image=str(LOGO))
    st.html(SUITE_CSS)
    st.sidebar.markdown(suite_menu_markdown())


def suite_menu_markdown():
    """The farm app suite's app list, as sidebar text: a link for each live app."""
    apps = json.loads(APPS_FILE.read_text(encoding="utf-8"))["apps"]
    lines = ["**Our farm apps**", ""]
    for app in apps:
        if app["id"] == APP_ID:
            lines.append(f"- {app['name']} (you're here)")
        elif app["url"]:
            lines.append(f"- [{app['name']}]({app['url']})")
        else:
            lines.append(f"- {app['name']} (coming soon)")
    return "\n".join(lines)
