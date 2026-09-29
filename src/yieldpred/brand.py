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

from yieldpred.crops import CROPS, DEFAULT_CROP, get_crop

ASSETS = Path(__file__).resolve().parents[2] / "app" / "assets"
LOGO = ASSETS / "logo.png"
PAGE_ICON = str(LOGO)
SUITE_CSS_FILE = ASSETS / "suite.css"    # copy of herd-planner/brand/suite.css
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

# The states the app can show, and the FIPS code each one's geometry is filed
# under. Every loader in appdata.py already took a `state` argument; what was
# missing was any way for a reader to change it, which made the header's
# "Nebraska & Iowa" a promise the app didn't keep.
STATES = {"ne": ("Nebraska", "31"), "ia": ("Iowa", "19")}
DEFAULT_STATE = "ne"

# Pages whose content actually changes with the state. How It Works is static,
# and Does It Transfer? is *about* both states at once - showing an inert state
# toggle on those two would be worse than not showing one, because a control
# that does nothing when you click it is how an app teaches you not to trust it.
STATE_AWARE = {"Home", "Maps", "County Explorer", "Model & Validation"}
# Pages with a Corn / Soybeans switch. Does It Transfer? has one too: for corn it asks "does a Nebraska
# model predict Iowa?", for soybeans "does the corn model help predict soybeans?".
CROP_AWARE = STATE_AWARE | {"Does It Transfer?"}

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
NAV_PULL_PX = BLOCK_PADDING_TOP_PX - HEADER_HEIGHT_PX

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
    # No feature list here: it used to read "weather, irrigation and soils",
    # which stopped being true the moment Iowa became reachable.
    "Home": "Predicting county corn yields — and being honest about how well that works.",
    "How It Works": "Where do the numbers come from?",
    "Maps": "Where does corn do well — and where is the model wrong?",
    "County Explorer": "What happened in one county, year by year?",
    "Model & Validation": "How much should any of this be trusted?",
    "Does It Transfer?": "Does a model built on Nebraska work anywhere else?",
}

# The reading order the menu numbers. Home is the cover, not a step.
STEPS = {label: i for i, (_p, label, _ic, _u) in enumerate(NAV) if label != "Home"}

# A few words under each step in the menu, so the menu says what each page answers. (Home's
# old "Start here" cards did this for three pages; since 2026-09-30 the menu is the one step bar,
# on every page, so its numbers and names always match.) Written for corn; for_crop() adapts them.
NAV_BLURBS = {
    "How It Works": "Where the numbers come from.",
    "Maps": "Where corn does well, and where the model misses.",
    "County Explorer": "One county, year by year.",
    "Model & Validation": "How much to trust it.",
    "Does It Transfer?": "Other states, other crops.",
}

def shared_footer_css() -> str:
    """The deep footer's rules, read out of the shared suite.css.

    Every other app links suite.css with a style tag. Streamlit cannot: its
    `st.html` sanitizer strips a link element outright - measured by serving a
    stylesheet and watching the tag never reach the DOM - so this app would
    otherwise have to keep its own copy of the footer rules and let them drift
    from the other three, which is exactly the problem this was meant to fix.

    So it reads the shared file and inlines the marked section instead. The app
    uses the same rules as everyone else; only the delivery differs. The markers
    are in suite.css, and a test fails if they go missing or the section turns up
    empty.
    """
    text = SUITE_CSS_FILE.read_text(encoding="utf-8")
    try:
        after = text.split(FOOTER_MARK_START, 1)[1]
        return after.split(FOOTER_MARK_END, 1)[0]
    except IndexError as err:                        # pragma: no cover - guarded by a test
        raise ValueError(
            f"{SUITE_CSS_FILE.name} has no {FOOTER_MARK_START} / {FOOTER_MARK_END} "
            "markers; the shared footer rules cannot be found") from err


FOOTER_MARK_START = "/* == suite-footer:start == */"
FOOTER_MARK_END = "/* == suite-footer:end == */"

# The pieces of the suite look (app/assets/suite.css) that .streamlit/config.toml
# can't set. Streamlit's own element names (data-testid) can change between
# versions; if the header turns white again after an upgrade, check them here.
SHARED_FOOTER_CSS = shared_footer_css()

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
  height: {HEADER_HEIGHT_PX}px; display: flex; align-items: center; gap: 14px;
  z-index: 999991; pointer-events: none; color: #fff;
}}
/* Suite version 6: the company ABOVE the app's name, as in every app in the suite (a hierarchy, like a
   series name over a book title), with the app's own logo - Streamlit's, at x=16..48 - standing to the
   left of both lines. */
.suite-headerbar .titles {{ display: flex; flex-direction: column; line-height: 1.15; }}
.suite-headerbar .name {{ font-weight: 700; font-size: 1.15rem; letter-spacing: .2px; }}
/* The way back to the company site (the suite's .suite-company). The bar ignores clicks so Streamlit's
   own header keeps working; this one link takes them back. */
.suite-headerbar a.company {{ pointer-events: auto; color: rgba(255, 255, 255, .78); font-weight: 600;
  font-size: .75rem; letter-spacing: .02em; text-decoration: none; white-space: nowrap; }}
.suite-headerbar a.company:hover {{ color: #fff; text-decoration: underline; text-underline-offset: 3px; }}
.suite-headerbar a.company::after {{ content: " ›"; }}
.suite-headerbar .ctx {{ font-size: .95rem; opacity: .9; }}
/* The app's name and Home both go to the home page. Home is a small outlined
   button after the context; on the home page it gets the same flat gold bar
   the step cards use for the current page. */
.suite-headerbar a.name {{ pointer-events: auto; color: #fff; text-decoration: none; }}
.suite-headerbar a.home {{
  pointer-events: auto; position: relative; color: #fff; text-decoration: none;
  font-weight: 600; font-size: .9rem; padding: 4px 12px;
  border: 1px solid rgba(255, 255, 255, .45); border-radius: 999px; white-space: nowrap;
}}
.suite-headerbar a.home:hover {{ border-color: {GOLD}; }}
.suite-headerbar a.home.active::after {{
  content: ""; position: absolute; left: 10px; right: 10px; bottom: -7px;
  height: 3px; border-radius: 2px; background: {GOLD};
}}

/* ---- vertical rhythm --------------------------------------------------
   Streamlit's defaults leave a lot of air, and on a page that is mostly
   a title, two controls and a figure it reads as an empty screen. These
   pull the title up under the menu band and close the gap between a
   heading and the thing it labels, without cramping body text. */
[data-testid="stHeading"] h1 {{
  font-size: 1.6rem; font-weight: 700; letter-spacing: .1px;
  padding-top: 0; margin-bottom: .15rem;
}}
/* Streamlit's defaults made h2 larger than the h1 above once the h1 came
   down to Herd Planner's scale. Note `st.subheader` renders an h3, not an
   h2 - so h3 is a section heading here and has to stay comfortably bigger
   than body text. Measured scale: 25.6 / 22.4 / 20 px. */
[data-testid="stHeading"] h2 {{
  font-size: 1.4rem; font-weight: 700; padding-top: 1rem; margin-bottom: .15rem;
}}
[data-testid="stHeading"] h3 {{
  font-size: 1.25rem; font-weight: 700; padding-top: 1rem; margin-bottom: .2rem;
}}
.suite-question {{
  color: #5e5c57; font-size: 1rem; margin: 0 0 1.2rem; max-width: 78ch;
}}

/* Below this the Deploy button and the context line would collide. */
@media (max-width: 760px) {{ .suite-headerbar .ctx {{ display: none; }} }}

/* ---- the menu row -----------------------------------------------------
   On the paper, below the green bar - which is where Herd Planner puts
   its step pills. An earlier version had them inside a green band flush
   against the header, which made the header look like a two-storey bar
   rather than a header with a page under it.

   NOTE: no angle brackets anywhere in this stylesheet, not even inside a
   comment. `st.html` runs its argument through an HTML sanitizer, which
   reads the text inside a style element as markup. An earlier version of
   this comment named a class with its key in angle brackets; the
   sanitizer read that as an open tag and silently discarded THE ENTIRE
   STYLESHEET - no error, no warning, just an unstyled app. The test
   `test_the_stylesheet_has_no_angle_brackets` guards it now. */
.suite-nav {{
  margin: -{NAV_PULL_PX}px 0 1.4rem;
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
}}

/* The step cards: the numbered pages 1 to 5 (Home is in the green header), one card each with a few
   words on what the page answers. This is the only menu (it replaced a pill
   row plus Home's three Start Here cards, whose numbers didn't match).
   Every card is white with a gold number circle (mockup A). The current page
   keeps the white card, gets an inverted number circle, and a flat gold bar
   sits in the gap just below it, like a pointer (option C, Scott, 2026-09-30:
   a whole gold card was overwhelming and a shadowed bar looked dated). */
.suite-steps {{
  display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px 10px; width: 100%;
}}
.suite-steps .step {{
  position: relative; display: block; background: #fff; border: 1px solid #e2e0d9; border-radius: 10px;
  padding: 12px 14px 14px; text-decoration: none; color: #1d1c1a;
  transition: border-color .12s;
}}
.suite-steps .step:hover {{ border-color: #b8862b; }}
.suite-steps .t {{
  display: flex; align-items: center; gap: 8px;
  font-weight: 700; font-size: 1rem; line-height: 1.2; color: {DEEP_GREEN};
}}
.suite-steps .n {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%; flex: none;
  background: {GOLD}; color: {DEEP_GREEN}; font-size: .78rem; font-weight: 700;
}}
.suite-steps .d {{ margin: 6px 0 0; font-size: .85rem; line-height: 1.35; color: #5e5c57; }}
.suite-steps .step.active::after {{
  content: ""; position: absolute; left: 10px; right: 10px; bottom: -9px;
  height: 4px; border-radius: 2px; background: {GOLD};
}}
.suite-steps .step.active .n {{ background: {DEEP_GREEN}; color: #fff; }}

/* The state switch, at the right end of the row. Squarer than the page
   pills on purpose: it is a setting, not a destination, and two controls
   that look identical but behave differently is worse than two that look
   different. Its active state is deep green rather than gold, so the row
   never shows two gold controls meaning two different things. */
.suite-views {{ margin-left: auto; display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
.suite-state {{ display: flex; gap: 0; }}
.suite-state .spill {{
  border: 1px solid #e2e0d9; border-right-width: 0; background: #fff;
  padding: 6px 14px; text-decoration: none; color: #5e5c57;
  font-weight: 600; font-size: .88rem;
}}
.suite-state .spill:first-child {{ border-radius: 8px 0 0 8px; }}
.suite-state .spill:last-child {{ border-radius: 0 8px 8px 0; border-right-width: 1px; }}
.suite-state .spill:hover {{ background: #fdfcf8; }}
.suite-state .spill.active {{
  background: {DEEP_GREEN}; color: #fff; border-color: {DEEP_GREEN};
}}

/* Between phone and desktop: three cards a row. On phones: two a row, names only,
   so the menu doesn't fill the first screen. */
@media (max-width: 1100px) {{ .suite-steps {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
@media (max-width: 760px) {{
  .suite-steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 6px; }}
  .suite-steps .step.active::after {{ bottom: -8px; height: 3px; }}

  .suite-steps .step {{ padding: 8px 10px 10px; }}
  .suite-steps .t {{ font-size: .92rem; }}
  .suite-steps .d {{ display: none; }}
  .suite-views {{ margin-left: 0; }}
}}

/* ---- the footer -------------------------------------------------------
   The rules themselves come from the shared suite.css, inlined above by
   shared_footer_css(). Only the two lines Streamlit needs live here: the
   other apps put the footer at the end of an ordinary page, while this one
   sits inside a centred block container, so the band is pulled back out to
   the window edges with the 50vw trick. */
{SHARED_FOOTER_CSS}
.site-footer {{ margin: 2.5rem calc(50% - 50vw) 0; }}

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


def current_state() -> str:
    """Which state the reader is looking at, from the URL.

    A query parameter rather than `st.session_state`, for a reason the pills
    forced: they are plain anchors, so following one is a full page load and
    starts a *new* Streamlit session - session state would be wiped on every
    click. The URL survives that, and it also makes a view shareable and
    bookmarkable, which session state never was.
    """
    import streamlit as st

    try:
        value = st.query_params.get("state", DEFAULT_STATE)
    except Exception:                      # AppTest has no query params
        return DEFAULT_STATE
    return value if value in STATES else DEFAULT_STATE


def current_crop() -> str:
    """Which crop the reader is looking at, from the URL (?crop=soybeans), like current_state.
    Corn, the default, keeps every existing address unchanged."""
    import streamlit as st

    try:
        value = st.query_params.get("crop", DEFAULT_CROP)
    except Exception:                      # AppTest has no query params
        return DEFAULT_CROP
    return value if value in CROPS else DEFAULT_CROP


def view_query(state: str = DEFAULT_STATE, crop: str = DEFAULT_CROP) -> str:
    """The query string that keeps the reader's state and crop when they follow a link.
    Defaults are left out, so Nebraska corn keeps the plain addresses it always had."""
    parts = ([f"state={state}"] if state != DEFAULT_STATE else []) + \
            ([f"crop={crop}"] if crop != DEFAULT_CROP else [])
    return "?" + "&".join(parts) if parts else ""


def crop_name(crop: str = DEFAULT_CROP) -> str:
    return get_crop(crop).name


def crop_word(crop: str = DEFAULT_CROP) -> str:
    """The crop inside a sentence: "corn yields", "soybean yields"."""
    return get_crop(crop).word


def for_crop(text: str, crop: str = DEFAULT_CROP) -> str:
    """Fixed wording written for corn (questions, cards), said about the crop being shown."""
    if crop == DEFAULT_CROP:
        return text
    plural = crop_name(crop).lower()                   # "corn does well" -> "soybeans do well"
    return (text.replace("does corn do", f"do {plural} do").replace("corn does", f"{plural} do")
            .replace("corn", crop_word(crop)))


def state_name(state: str) -> str:
    return STATES[state][0]


def state_fips(state: str) -> str:
    return STATES[state][1]


def page_setup(page_title: str, *, layout: str = "wide") -> str:
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
    state, crop = current_state(), current_crop()
    header_bar(state, crop, current=page_title)
    nav_bar(current=page_title, state=state, crop=crop)
    return state


def page_heading(page_title: str, display: str | None = None,
                 show_title: bool = True) -> None:
    """The page's title and the question it answers.

    The question comes from one dict in this module rather than from six page
    files, so the menu, the tab title and the heading can't drift apart.

    `display` is for a page whose menu label and heading differ.

    `show_title=False` drops the heading and keeps only the question. The home
    page uses it: the green bar already says "Yield Predictor - Corn, Nebraska",
    so a display headline underneath repeated the header in larger type.
    """
    import streamlit as st

    if show_title:
        st.title(display or page_title)
    question = QUESTIONS.get(page_title)
    if question:
        question = for_crop(question, current_crop())
        st.html(f'<p class="suite-question">{question}</p>')


def header_bar(state: str = DEFAULT_STATE, crop: str = DEFAULT_CROP, current: str | None = None) -> None:
    """The app name and context, in white, beside the logo in the green bar.

    This is the piece Herd Planner has and this app was missing - its header
    reads "Herd Planner   Ranch: Demo Ranch (demo)" and ours was a logo and
    empty green. Same shape here: `h1`-weight name, then a lighter context line
    at 0.95rem and 0.9 opacity, exactly matching `header.top .ranch` in
    herd-planner/src/herd_planner/web/style.css.
    """
    import streamlit as st

    st.html(header_bar_html(state, crop, current))


def header_bar_html(state: str = DEFAULT_STATE, crop: str = DEFAULT_CROP, current: str | None = None) -> str:
    """The header bar's HTML: the company link above the app's name (back to the company site, like
    every app in the suite), then the context. target="_top" because Streamlit Community Cloud
    shows the app inside a frame, and the company site should replace the whole page, not the frame."""
    company = load_company()
    # Home lives in the green header (Scott, 2026-09-30), so the step cards are just 1 to 5.
    home_on = " active" if current == "Home" else ""
    home_attr = ' aria-current="page"' if current == "Home" else ""
    back = ""
    if company:
        back = f'<a class="company" href="{company["url"]}" target="_top">{company["name"]}</a>'
    return f"""
    <div class="suite-headerbar">
      <span class="titles">{back}<a class="name" href="./{view_query(state, crop)}" target="_self">{APP_NAME}</a></span>
      <span class="ctx">{state_name(state)} {crop_name(crop)}</span>
      <a class="home{home_on}" href="./{view_query(state, crop)}" target="_self"{home_attr}>Home</a>
    </div>
    """


def nav_bar(current: str | None = None, state: str = DEFAULT_STATE, crop: str = DEFAULT_CROP) -> None:
    """The menu: one card per page (Home, then the numbered steps), each with a few words on what
    the page answers, the current one gold. Since 2026-09-30 this is the only step bar; it replaced
    a pill row plus three "Start here" cards on Home whose numbers didn't match the menu's.

    The numbering follows Herd Planner's step pills ("1 Your ranch / 2 Your cattle / ..."): a
    number in a circle, the current one gold. The same idea works
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

    query = view_query(state, crop)
    cards = []
    for _path, label, _icon, url in NAV:
        step = STEPS.get(label)
        if not step:
            continue  # Home is in the green header, not a step
        number = f'<span class="n">{step}</span>'
        active = " active" if label == current else ""
        here_attr = ' aria-current="page"' if label == current else ""
        cards.append(f'<a class="step{active}" href="{url or "./"}{query}"{here_attr}>'
                     f'<span class="t">{number}<span class="l">{label}</span></span>'
                     f'<p class="d">{for_crop(NAV_BLURBS[label], crop)}</p></a>')

    here = next((u for _p, lab, _i, u in NAV if lab == current), "") or "./"
    switch = ""
    # State first, then crop, so the pair reads "Nebraska Corn" (Scott, 2026-09-30), the way a farmer
    # says it. The state switch keeps the crop, and the crop switch keeps the state.
    if current in STATE_AWARE:
        switch += '<div class="suite-state">' + "".join(
            f'<a class="spill{" active" if code == state else ""}" href="{here}{view_query(code, crop)}">'
            f'{name}</a>' for code, (name, _fips) in STATES.items()) + "</div>"
    if current in CROP_AWARE:
        switch += '<div class="suite-state suite-crop">' + "".join(
            f'<a class="spill{" active" if key == crop else ""}" href="{here}{view_query(state, key)}">'
            f'{c.name}</a>' for key, c in CROPS.items()) + "</div>"

    # The switches above the cards, on the right; then the one step bar.
    views = f'<div class="suite-views">{switch}</div>' if switch else ""
    st.html(f'<div class="suite-nav">{views}<nav class="suite-steps" aria-label="Pages">{"".join(cards)}</nav></div>')


def load_suite_apps() -> list[dict]:
    """The farm app suite's app list, read from the shared JSON.

    One reader for two renderers - the markdown one and the footer's HTML one -
    so a new app appears in both by editing the JSON and nothing else.
    """
    return json.loads(APPS_FILE.read_text(encoding="utf-8"))["apps"]


def load_company() -> dict | None:
    """The company site every app links back to: "company" in the shared suite-apps.json."""
    return json.loads(APPS_FILE.read_text(encoding="utf-8")).get("company")


def load_account() -> dict | None:
    """The suite's account page (farm-account): "account" in the shared suite-apps.json."""
    return json.loads(APPS_FILE.read_text(encoding="utf-8")).get("account")


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


# What this app is, in one line, for the footer's left column. The one piece
# of the footer that is deliberately product-specific, alongside the Tools
# column - Farm Equipment Planner's reads "Is that machine worth it for your
# farm? ..." in the same slot.
FOOTER_DESCRIPTION = ("County corn and soybean yields from weather, soils and irrigation — "
                      "and an honest account of how far each number can be trusted.")

# The sources behind every number, as the footer's third column. Farm Equipment
# Planner uses that slot for Data partners; this app has no partners, it has
# public agencies, so the slot says where the data came from instead.
DATA_SOURCES_LINKS = [
    ("USDA NASS Quick Stats", "https://quickstats.nass.usda.gov/"),
    ("NASA POWER", "https://power.larc.nasa.gov/"),
    ("USDA Soil Data Access", "https://sdmdataaccess.sc.egov.usda.gov/"),
    ("USGS Elevation", "https://apps.nationalmap.gov/epqs/"),
    ("US Census Bureau", "https://www.census.gov/geographies.html"),
]


def _logo_data_uri() -> str:
    """The corn logo as a data URI.

    Streamlit serves `st.logo`'s image from a URL it makes up at runtime, and
    there is no supported way to ask it for that URL - so the footer embeds its
    own copy rather than guessing. Read once per process; it is 22 KB.
    """
    import base64
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def encode() -> str:
        return base64.b64encode(LOGO.read_bytes()).decode("ascii")

    return f"data:image/png;base64,{encode()}"


def page_footer(state: str = DEFAULT_STATE, crop: str | None = None) -> None:
    """The deep footer, matching Farm Equipment Planner's column for column.

    The shape is shared across the suite on purpose: brand and description on
    the left, then Tools / Farm apps / Data sources / About, a rule, then the
    disclaimer line and the copyright. What differs between apps is only what
    the product actually has.

    Two differences from Farm Equipment Planner, both deliberate:

    * Its second column is "Your account". This app has no accounts, so that
      slot carries the suite's other apps instead - the cross-app links that
      used to sit in the sidebar.
    * Its About column links Terms of Use and Privacy Policy. Those pages do
      not exist for this app, and a footer link that 404s is worse than a
      shorter column, so they are left out until the company site hosts one
      set of them for all three apps to point at.

    The legal notice itself stays a `st.caption(FOOTER)` call on paper above
    this band - it carries the typical-error figure, and tests/test_app_smoke.py
    checks every page renders it.
    """
    import streamlit as st

    crop = crop or current_crop()
    tools = "".join(
        f'<a href="{url or "./"}{view_query(state, crop) if url else ""}">{label}</a>'
        for _p, label, _i, url in NAV if label != "Home")

    apps = "".join(
        f'<a href="{app["url"]}" target="_blank" rel="noopener">{app["name"]}</a>'
        if app["url"] and app["id"] != APP_ID
        else f'<a href="./">{app["name"]}</a>' if app["id"] == APP_ID
        else f'<a class="soon" aria-disabled="true">{app["name"]} (coming soon)</a>'
        for app in load_suite_apps())
    # The suite account, last in the Farm apps column (the other apps put it at the top of their Farm Apps
    # menu; this app has no menu). target="_top": Streamlit Cloud shows the app inside a frame.
    account = load_account()
    if account:
        apps += f'<a href="{account["url"]}" target="_top">{account["name"]}</a>'

    sources = "".join(
        f'<a href="{url}" target="_blank" rel="noopener">{label}</a>'
        for label, url in DATA_SOURCES_LINKS)

    st.html(f"""
    <footer class="site-footer">
      <div class="footer-inner">
        <div class="footer-brand">
          <a class="brand" href="./"><img src="{_logo_data_uri()}" alt="" class="logo">{APP_NAME}</a>
          <p>{FOOTER_DESCRIPTION}</p>
        </div>
        <nav class="footer-col" aria-label="Tools">
          <h2>Tools</h2>
          {tools}
        </nav>
        <nav class="footer-col" aria-label="Farm apps">
          <h2>Farm apps</h2>
          {apps}
        </nav>
        <nav class="footer-col" aria-label="Data sources">
          <h2>Data sources</h2>
          {sources}
        </nav>
        <nav class="footer-col" aria-label="About">
          <h2>About</h2>
          <a href="How_It_Works">Disclaimer</a>
          <a href="How_It_Works">How it works</a>
        </nav>
      </div>
      <div class="footer-bottom">
        <p>Estimates for planning only, not agronomic, financial or insurance advice.
           <a href="How_It_Works">Read the disclaimer</a>.</p>
        <p>&copy; 2026 {APP_NAME} &middot; a data science learning project</p>
      </div>
    </footer>
    """)


def show_logo() -> None:
    """Deprecated: page_setup() does this and the rest of the chrome.

    Kept so that a page which hasn't been converted still gets the logo and the
    colors rather than silently losing them.
    """
    import streamlit as st

    st.logo(str(LOGO), size="large", icon_image=str(LOGO))
    st.html(SUITE_CSS)


