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
APP_ID = "corn-yield-predictor"          # this app's id in that list - an internal
                                         # key, deliberately not renamed with the
                                         # display name; churning ids buys nothing

# The product name. Parallel to "Herd Planner", the sibling already in the suite.
# "Nebraska Corn Yield Predictor" was wrong on both counts by 9/25: Iowa is in the
# app, and corn is about to stop being alone (see claude/crop-scope.md). The repo
# and folder are still named nebraska-yield-predictor and get renamed when the repo
# goes private, so that Streamlit Cloud is re-pointed once rather than twice.
APP_NAME = "Yield Predictor"

# The line beside the name in the header bar, matching Herd Planner's
# "Ranch: Demo Ranch (demo)". It says what this instance is showing.
APP_CONTEXT = "Corn · Nebraska & Iowa"

# The suite's deep green, the one value the header, the footer and suite.css all
# have to agree on. suite.css declares it as --suite-deep-green; Streamlit's own
# elements are styled from here because config.toml has no setting for them.
#
# Contrast on this green, computed rather than eyeballed (see the test):
#   white  #ffffff on #243b2f -> 12.6:1   (WCAG AAA)
#   gold   #d9a441 behind deep-green text -> 5.6:1 (AA)
# White on gold is only 2.2:1, which is why the active menu item is gold with
# deep-green text and never white text.
DEEP_GREEN = "#243b2f"
GOLD = "#d9a441"

# Streamlit's own geometry, measured in a browser rather than guessed, because
# the header bar and the menu band have to line up against it exactly:
#   header[data-testid="stHeader"]      60px tall, z-index 999990
#   [data-testid="stHeaderLogo"]        x = 16..48
#   [data-testid="stMainBlockContainer"] padding-top: 96px
# If a Streamlit upgrade changes any of these the menu band detaches from the
# header by a visible gap - which is a cosmetic failure, not a broken page.
HEADER_HEIGHT_PX = 60
BLOCK_PADDING_TOP_PX = 96

# How far up the menu band is pulled so it butts against the header with no
# seam. It is the block container's top padding, minus the header height, plus
# the 16px the two zero-height st.html elements above it contribute to the flow
# (the CSS block and the header-bar block). Measured, not derived: the 16px is
# not a flex gap on the parent, so there is nothing to read it from.
NAV_PULL_PX = BLOCK_PADDING_TOP_PX - HEADER_HEIGHT_PX + 16

# Every page, in menu order: (path, label, icon, url).
#
# `path` is relative to the entrypoint's directory, which is what st.page_link
# expects no matter which page is calling it.
#
# `url` is the address Streamlit serves that page at - the filename with its
# sort prefix and extension stripped, and "" for the home page. It is written
# out rather than derived because it is the hook the current-page highlight
# uses, and a wrong one fails silently by highlighting nothing. A test derives
# it from the filenames and checks it against this list.
#
# Menu order is the reader's order, not the repo's. "How It Works" comes second
# because the explanation was buried at position four, and first place belongs
# to the page that shows the app doing its job rather than to its manual.
NAV = [
    ("streamlit_app.py", "Home", ":material/home:", ""),
    ("pages/4_How_It_Works.py", "How It Works", ":material/build:", "How_It_Works"),
    ("pages/1_Maps.py", "Maps", ":material/map:", "Maps"),
    ("pages/2_County_Explorer.py", "County Explorer", ":material/search:", "County_Explorer"),
    ("pages/3_Model_and_Validation.py", "Model & Validation", ":material/insights:",
     "Model_and_Validation"),
    ("pages/5_Does_It_Transfer.py", "Does It Transfer?", ":material/swap_horiz:",
     "Does_It_Transfer"),
]

# What each page answers, shown as a line under its title. The app's pages were
# named after the things that were built - Maps, Model & Validation - which is
# the repo's structure, not a reader's. A page that states its question tells
# you whether you want to be on it, which is most of what a "flow" is.
QUESTIONS = {
    "Home": "Predicting county corn yields from weather, irrigation and soils — and being honest about how well that works.",
    "How It Works": "Where do the numbers come from?",
    "Maps": "Where does corn do well — and where is the model wrong?",
    "County Explorer": "What happened in one county, year by year?",
    "Model & Validation": "How much should any of this be trusted?",
    "Does It Transfer?": "Does a model built on Nebraska work anywhere else?",
}

# The reading order the menu numbers. Home is the cover, not a step.
STEPS = {label: i for i, (_p, label, _ic, _u) in enumerate(NAV) if label != "Home"}

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

/* ---- the name beside the logo -----------------------------------------
   Streamlit gives no API for putting content in its header, so this is a
   real element of ours, fixed into the header's 60px and sitting just
   right of the logo (which measures x=16..48). Real HTML rather than a
   CSS `content:` string, so it is selectable, translatable and visible to
   a test. `pointer-events: none` keeps it from swallowing clicks meant
   for the header underneath. */
.suite-headerbar {{
  position: fixed; top: env(safe-area-inset-top, 0px); left: 60px;
  height: {HEADER_HEIGHT_PX}px; display: flex; align-items: baseline; gap: 14px;
  z-index: 999991; pointer-events: none; color: #fff;
  padding-top: 17px;
}}
.suite-headerbar .name {{ font-weight: 700; font-size: 1.15rem; letter-spacing: .2px; }}
.suite-headerbar .ctx {{ font-size: .95rem; opacity: .9; }}

/* ---- vertical rhythm --------------------------------------------------
   Streamlit's defaults leave a lot of air, and on a page that is mostly
   a title, two controls and a figure it reads as an empty screen. These
   pull the title up under the menu band and close the gap between a
   heading and the thing it labels, without cramping body text. */
[data-testid="stHeading"] h1 {{ padding-top: .25rem; margin-bottom: .1rem; }}
[data-testid="stHeading"] h2 {{ padding-top: .75rem; margin-bottom: .1rem; }}
[data-testid="stHeading"] h3 {{ padding-top: .5rem; margin-bottom: .1rem; }}
.suite-question {{
  color: #5e5c57; font-size: 1rem; margin: 0 0 1.1rem; max-width: 62ch;
}}

/* ---- the Start Here strip ---------------------------------------------
   Three cards in a suggested reading order. Steps told honestly: this app
   takes no input, so a numbered wizard would be five clicks pretending to
   be a form. A reading order is a real thing to offer. */
.suite-start {{
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 14px; margin: .25rem 0 1.75rem;
}}
.suite-start a {{
  display: block; background: #fff; border: 1px solid #e2e0d9;
  border-radius: 10px; padding: 14px 16px; text-decoration: none;
}}
.suite-start a:hover {{ border-color: #b8862b; background: #fdfcf8; }}
.suite-start .n {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%;
  background: {GOLD}; color: {DEEP_GREEN};
  font-size: .8rem; font-weight: 700; margin-right: 8px;
}}
.suite-start .t {{ font-weight: 700; color: {DEEP_GREEN}; }}
.suite-start .d {{ color: #5e5c57; font-size: .9rem; margin: 6px 0 0; }}
@media (max-width: 760px) {{ .suite-start {{ grid-template-columns: 1fr; }} }}
/* Below this the Deploy button and the context line would collide. */
@media (max-width: 760px) {{ .suite-headerbar .ctx {{ display: none; }} }}

/* ---- the menu band ----------------------------------------------------
   The menu sits in a green band pulled up against the header, so the two
   read as one bar the way Herd Planner's does. The pull is the block
   container's top padding minus the header height, plus the flow space
   the zero-height html blocks above it take.

   NOTE: no angle brackets anywhere in this stylesheet, not even inside a
   comment. `st.html` runs its argument through an HTML sanitizer, which
   reads the text inside a style element as markup. An earlier version of
   this comment named a class with its key in angle brackets; the
   sanitizer read that as an open tag and silently discarded THE ENTIRE
   STYLESHEET - no error, no warning, just an unstyled app. The test
   `test_the_stylesheet_has_no_angle_brackets` guards it now. */
.suite-nav {{
  background: {DEEP_GREEN};
  margin: -{NAV_PULL_PX}px calc(50% - 50vw) 1.5rem;
  padding: .6rem calc(50vw - 50%) .7rem;
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
}}

/* The pills. Herd Planner's shape: fully rounded, a number in a circle,
   the current one gold. Inactive pills are outlined rather than filled,
   because six filled pills on green is a lot of boxes. */
.suite-nav .pill {{
  display: inline-flex; align-items: center; gap: 8px;
  border: 1px solid rgba(255, 255, 255, .35); border-radius: 999px;
  padding: 6px 14px 6px 8px; text-decoration: none;
  color: #fff; font-weight: 600; font-size: .92rem; line-height: 1.2;
  background: transparent; transition: background .12s, border-color .12s;
}}
.suite-nav .pill:hover {{
  background: rgba(255, 255, 255, .14); border-color: rgba(255, 255, 255, .6);
}}
.suite-nav .pill .n {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%; flex: none;
  background: rgba(255, 255, 255, .18); color: #fff;
  font-size: .78rem; font-weight: 700;
}}
/* Home has no number, so it needs the padding a number would have given. */
.suite-nav .pill:not(:has(.n)) {{ padding-left: 14px; }}

/* The current page: gold, with deep-green text. Never white on gold -
   that is 2.2:1. The number circle inverts to deep green on gold, which
   is how Herd Planner marks its active step. */
.suite-nav .pill.active {{
  background: {GOLD}; border-color: {GOLD}; color: {DEEP_GREEN};
}}
.suite-nav .pill.active:hover {{ background: #e6b75a; border-color: #e6b75a; }}
.suite-nav .pill.active .n {{ background: {DEEP_GREEN}; color: #fff; }}

@media (max-width: 760px) {{
  .suite-nav .pill {{ font-size: .86rem; padding: 5px 11px 5px 6px; }}
}}

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

    The browser tab reads "<page> · Yield Predictor", so a pinned tab or a
    window switcher says which app it is, not just which page.
    """
    import streamlit as st  # imported here so the pipeline can import yieldpred without Streamlit

    tab_title = APP_NAME if page_title == APP_NAME else f"{page_title} · {APP_NAME}"
    st.set_page_config(page_title=tab_title, page_icon=PAGE_ICON, layout=layout,
                       initial_sidebar_state="collapsed")
    st.logo(str(LOGO), size="large", icon_image=str(LOGO))
    st.html(SUITE_CSS)
    header_bar()
    nav_bar(current=page_title)


def page_heading(page_title: str, display: str | None = None) -> None:
    """The page's title and the question it answers.

    The question comes from one dict in this module rather than from six page
    files, so the menu, the tab title and the heading can't drift apart.

    `display` is for the home page, whose menu label is "Home" but whose
    heading should say what the page is actually about.
    """
    import streamlit as st

    st.title(display or page_title)
    question = QUESTIONS.get(page_title)
    if question:
        st.html(f'<p class="suite-question">{question}</p>')


def header_bar() -> None:
    """The app name and context, in white, beside the logo in the green bar.

    This is the piece Herd Planner has and this app was missing - its header
    reads "Herd Planner   Ranch: Demo Ranch (demo)" and ours was a logo and
    empty green. Same shape here: `h1`-weight name, then a lighter context line
    at 0.95rem and 0.9 opacity, exactly matching `header.top .ranch` in
    herd-planner/src/herd_planner/web/style.css.
    """
    import streamlit as st

    st.html(f"""
    <div class="suite-headerbar">
      <span class="name">{APP_NAME}</span>
      <span class="ctx">{APP_CONTEXT}</span>
    </div>
    """)


def nav_bar(current: str | None = None) -> None:
    """The menu, as Herd Planner's numbered step pills.

    Herd Planner shows "1 Your ranch / 2 Your cattle / ..." as rounded pills
    with the number in a circle, the current one gold. The same shape works
    here, with one honest difference worth stating: those are steps with state,
    and these are a *reading order*. Nothing has to be completed before
    anything else, and the pages can be visited in any order - the numbers say
    "this is the order it makes sense in", which is what a first-time reader
    was missing. Home carries no number because it is the cover, not a step.

    Plain anchors rather than `st.page_link`. Three reasons: the pill needs a
    number circle and a label as separate elements, which page_link's plain
    string label cannot express; page_link resolves its path against the
    *entrypoint*, which made the whole menu raise under
    `AppTest.from_file(one_page)` and needed a swallowed exception to survive;
    and the hrefs are the same page URLs Streamlit itself serves, carried
    explicitly in NAV and checked against the filenames by a test.

    Home's href is "./" and not "", because an empty href means "this page"
    and would make the Home pill a no-op from every other page.
    """
    import streamlit as st

    pills = []
    for _path, label, _icon, url in NAV:
        step = STEPS.get(label)
        number = f'<span class="n">{step}</span>' if step else ""
        state = " active" if label == current else ""
        pills.append(f'<a class="pill{state}" href="{url or "./"}">'
                     f'{number}<span class="l">{label}</span></a>')
    st.html(f'<div class="suite-nav">{"".join(pills)}</div>')


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
      <strong>{APP_NAME}</strong> — part of the First Light Ag farm app suite<br>
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


# The suggested reading order offered on the home page: (url, title, why).
START_HERE = [
    ("How_It_Works", "How It Works",
     "Where the numbers come from — the data, the model, the vocabulary."),
    ("Maps", "Maps",
     "Where corn does well, and where the model gets it wrong."),
    ("Model_and_Validation", "How Much To Trust It",
     "The same model scored four ways, and why the answers differ."),
]


def start_here() -> None:
    """A suggested reading order, as three cards on the home page.

    Deliberately not a wizard. A wizard implies the user supplies something at
    step one, and this app reads precomputed files - numbered steps over a
    read-only report would be ceremony, and a reader who clicked "1" expecting
    to enter their county would be more lost than before. What the app can
    honestly offer is an order to read it in.

    Plain anchors rather than `st.page_link`, because these need to be cards
    with a number, a title and a line of why; the hrefs are the page URLs
    Streamlit serves, the same ones NAV carries.
    """
    import streamlit as st

    cards = "".join(
        f'<a href="{url}"><div><span class="n">{i}</span>'
        f'<span class="t">{title}</span></div>'
        f'<p class="d">{why}</p></a>'
        for i, (url, title, why) in enumerate(START_HERE, start=1))
    st.html(f'<div class="suite-start">{cards}</div>')
