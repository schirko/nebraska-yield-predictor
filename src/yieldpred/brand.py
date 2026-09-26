"""The app's logo, the farm app suite's shared look, and the page navigation.

The logo is an ear of corn on a soil-brown circle, in the same family as Herd
Planner's cow and Farm Equipment Planner's tractor: a colored circle, a cream
drawing, a wheat-gold accent. The master SVG is app/assets/logo.svg; logo.png is
rendered from it because Streamlit's tab icon and st.logo are most reliable with
a PNG.

WHY THE SIDEBAR IS GONE

Streamlit puts multipage navigation in a left sidebar by default. In this app
that sidebar was doing two bad things at once.

The first was visible. Streamlit lays the header out as a *sibling* of the
sidebar, not above it, so the green header bar started where the sidebar ended:

    header[data-testid="stHeader"]   x=300  width=1100   background #243b2f
    [data-testid="stSidebar"]        x=0    width=300    background #efece4

Every pixel of the top-left corner belonged to the sidebar, which is why the
green stopped a fifth of the way in. No amount of styling the header fixes that,
because the header was never there to style.

The second was structural. Herd Planner is a plain web app with a full-width
green bar and links across it. As long as this app kept a left rail, the two
could share colors and still not read as one product.

So the navigation moved into the page body and the sidebar is hidden. The header
then spans the window on its own, with no width override needed - the fix for
the look and the fix for the layout are the same fix.

The pages/ directory stays exactly as it was. `st.navigation(position="top")` is
the other way to do this, and it is the *supported* way, but it means giving up
file-based pages, and every page-level test in tests/test_app_smoke.py runs
`AppTest.from_file()` against one page at a time. Keeping the files keeps the
tests, and a row of `st.page_link` calls is navigation in the body either way.

Every page starts with page_setup() and ends with page_footer(), so the chrome
lives in one file and six pages cannot drift apart.

WHERE THE "OUR FARM APPS" LIST WENT

It used to be the last thing in the sidebar. With the sidebar gone it moved into
the green footer band, which is also where Herd Planner keeps its cross-app
links - so the two apps now agree on the placement as well as the color. The
list itself still comes from app/assets/suite-apps.json, a copy of the master in
herd-planner/brand/, so adding an app to the suite is still one file in one
place.
"""

import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "app" / "assets"
LOGO = ASSETS / "logo.png"
PAGE_ICON = str(LOGO)
APPS_FILE = ASSETS / "suite-apps.json"   # copy of herd-planner/brand/suite-apps.json
APP_ID = "corn-yield-predictor"          # this app's id in that list

# The suite's deep green, the one value the header, the footer and suite.css all
# have to agree on. suite.css declares it as --suite-deep-green; Streamlit's own
# elements are styled from here because config.toml has no setting for them.
DEEP_GREEN = "#243b2f"

# Every page, in menu order. The path is relative to the entrypoint's directory,
# which is what st.page_link expects no matter which page is calling it.
NAV = [
    ("streamlit_app.py", "Home", ":material/home:"),
    ("pages/1_Maps.py", "Maps", ":material/map:"),
    ("pages/2_County_Explorer.py", "County Explorer", ":material/search:"),
    ("pages/3_Model_and_Validation.py", "Model & Validation", ":material/insights:"),
    ("pages/4_How_It_Works.py", "How It Works", ":material/build:"),
    ("pages/5_Does_It_Transfer.py", "Does It Transfer?", ":material/swap_horiz:"),
]

# The pieces of the suite look (app/assets/suite.css) that .streamlit/config.toml
# can't set. Streamlit's own element names (data-testid) can change between
# versions; if the header turns white again after an upgrade, check them here.
SUITE_CSS = f"""
<style>
/* ---- the header bar -------------------------------------------------- */
header[data-testid="stHeader"] {{ background: {DEEP_GREEN}; }}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a,
header[data-testid="stHeader"] [data-testid="stMainMenu"] * {{ color: #ffffff; }}

/* ---- the sidebar, which no longer exists ------------------------------
   The menu lives in the page body now. These names cover the rail itself
   and the controls Streamlit offers for reopening it; a version upgrade
   that renames one shows up as a stray arrow, not as a broken page. */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {{ display: none !important; }}

/* ---- the body navigation ---------------------------------------------- */
[data-testid="stPageLink"] a {{
  border-radius: 8px; padding: 6px 10px; font-weight: 600;
  color: {DEEP_GREEN}; text-decoration: none;
}}
[data-testid="stPageLink"] a:hover {{ background: #ece9e0; }}
[data-testid="stPageLink"] a p {{ font-weight: 600; }}

/* ---- the footer band --------------------------------------------------
   Full-bleed: the main block container is centred and padded, so the band
   is pulled back out to the window edges with the standard 50vw trick and
   its own padding puts the text back in line with the body text. */
.suite-footer {{
  background: {DEEP_GREEN}; color: rgba(255, 255, 255, .82);
  margin: 2.5rem calc(50% - 50vw) 0;
  padding: 1.5rem calc(50vw - 50%) 1.75rem;
  font-size: .85rem; line-height: 1.55;
}}
.suite-footer strong {{ color: #fff; font-weight: 700; }}
.suite-footer a {{ color: rgba(255, 255, 255, .82); }}
.suite-footer a:hover {{ color: #fff; }}

/* ---- widgets ----------------------------------------------------------
   Gold buttons take deep-green text: white on gold is too faint to read
   (2.2:1). The slider's value label would be gold on paper, likewise. */
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
  color: {DEEP_GREEN} !important; font-weight: 600;
}}
[data-testid="stSliderThumbValue"] {{ color: {DEEP_GREEN}; }}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  background: #e6b75a; border-color: #e6b75a;
}}
</style>
"""


def page_setup(page_title: str, *, layout: str = "wide") -> None:
    """The first Streamlit call on every page: config, logo, chrome, menu.

    `initial_sidebar_state="collapsed"` matters even though the sidebar is
    hidden by CSS. Streamlit decides *where to render the logo* from that
    state - expanded puts it inside the sidebar, which is display:none, so the
    logo would vanish along with the rail. Collapsed puts it in the header,
    which is where it belongs now.
    """
    import streamlit as st  # imported here so the pipeline can import yieldpred without Streamlit

    st.set_page_config(page_title=page_title, page_icon=PAGE_ICON, layout=layout,
                       initial_sidebar_state="collapsed")
    st.logo(str(LOGO), size="large", icon_image=str(LOGO))
    st.html(SUITE_CSS)
    nav_bar()


def nav_bar() -> None:
    """The menu, across the top of the page body.

    A horizontal container rather than `st.columns(len(NAV))`. Equal columns
    space six labels of very different lengths evenly and leave the short ones
    marooned in the middle of a wide gap; a wrapping flex row packs them from
    the left at their natural widths and folds onto a second line on a narrow
    window instead of squeezing.

    Streamlit marks the current page's link itself - `st.page_link` reads the
    active page from the runtime - so there is nothing to highlight here, and
    nothing that could disagree with the address bar.

    The `try` is not defensive noise; it is what lets a page be tested alone.
    `st.page_link` resolves its path against *the entrypoint file*, and raises
    `StreamlitPageNotFoundError` for anything the running app hasn't registered.
    Under the real server the entrypoint is always app/streamlit_app.py, so all
    six resolve. Under `AppTest.from_file("pages/1_Maps.py")` that page is the
    entrypoint and the only registered page, so the other five don't exist and
    the menu would take every page-level test down with it.

    Swallowing the error would normally be the wrong trade - a typo in NAV would
    make a link quietly disappear in production. It isn't one here, because
    `tests/test_app_smoke.py` asserts statically that NAV and app/pages/ name
    exactly the same files. The static check catches a wrong entry; this catch
    only ever fires for a page that genuinely isn't part of the app being run.
    """
    import streamlit as st
    from streamlit.errors import StreamlitPageNotFoundError

    with st.container(horizontal=True, wrap=True, gap="small"):
        for path, label, icon in NAV:
            try:
                st.page_link(path, label=label, icon=icon)
            except StreamlitPageNotFoundError:
                continue
    st.divider()


def load_suite_apps() -> list[dict]:
    """The farm app suite's app list, read from the shared JSON.

    One reader for two renderers - the markdown one and the footer's HTML one -
    so a new app appears in both by editing the JSON and nothing else.
    """
    return json.loads(APPS_FILE.read_text(encoding="utf-8"))["apps"]


def suite_menu_markdown() -> str:
    """The suite's app list as markdown: a link for each live app."""
    lines = ["**Our farm apps**", ""]
    for app in load_suite_apps():
        if app["id"] == APP_ID:
            lines.append(f"- {app['name']} (you're here)")
        elif app["url"]:
            lines.append(f"- [{app['name']}]({app['url']})")
        else:
            lines.append(f"- {app['name']} (coming soon)")
    return "\n".join(lines)


def _suite_menu_html() -> str:
    """The same list, inline, for the footer band."""
    parts = []
    for app in load_suite_apps():
        if app["id"] == APP_ID:
            parts.append(f"<strong>{app['name']}</strong> (you're here)")
        elif app["url"]:
            parts.append(f'<a href="{app["url"]}" target="_blank" '
                         f'rel="noopener">{app["name"]}</a>')
        else:
            parts.append(f"{app['name']} (coming soon)")
    return " · ".join(parts)


def page_footer() -> None:
    """The deep-green band that closes every page, matching Herd Planner's.

    The legal notice itself stays a `st.caption(FOOTER)` call in each page, on
    paper above this band, because it has to be plainly readable and because
    tests/test_app_smoke.py checks for that exact call in every page's source.
    This band carries the identity, the suite's other apps and the sources.
    """
    import streamlit as st

    st.html(f"""
    <div class="suite-footer">
      <strong>Corn Yield Predictor</strong> — part of the farm app suite<br>
      Our farm apps: {_suite_menu_html()}<br>
      Data: USDA NASS Quick Stats · NASA POWER · USDA Soil Data Access ·
      USGS · US Census Bureau.
      Built with Python, scikit-learn, GeoPandas and Streamlit.
    </div>
    """)


def show_logo() -> None:
    """Deprecated: page_setup() does this and the rest of the chrome.

    Kept so that a page which hasn't been converted still gets the logo and the
    colors rather than silently losing them.
    """
    import streamlit as st

    st.logo(str(LOGO), size="large", icon_image=str(LOGO))
    st.html(SUITE_CSS)
