"""Every page loads without raising.

Streamlit's `AppTest` runs a page the way the server would - top to bottom, in
process - and collects what it rendered. It catches the class of bug that unit
tests can't: a typo in a page, a renamed function the page still imports, a
column that no longer exists.

These tests pass whether or not the data files have been generated. With no data
each page shows its "run this script first" warning and stops; with data it
renders. Both are correct behaviour, and asserting only "no exception" is what
makes the test useful in a fresh clone.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "streamlit_app.py"] + sorted((APP / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_runs_without_exception(page):
    app = AppTest.from_file(str(page), default_timeout=120).run()
    assert not app.exception, f"{page.name}: {[e.value for e in app.exception]}"


def test_every_page_can_find_src_without_an_installed_package():
    """The app must not depend on `pip install -e .` having worked.

    On Streamlit Community Cloud the package build happens on a machine we don't
    control, and when it fails the app dies at its first `from yieldpred...` import
    with a traceback that points at the import line rather than at the cause. So
    every page puts src/ on sys.path itself, and this test checks that the shim is
    actually there and actually resolves - which a normal test run can't tell you,
    because locally the package IS installed and the shim never gets used.
    """
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        assert "sys.path.insert" in text, f"{page.name} has no src/ shim"

        # Recompute the path exactly as the page does: parents[1] for the home
        # page, parents[2] for anything under pages/.
        depth = 1 if page.parent.name == "app" else 2
        src = page.resolve().parents[depth] / "src"
        assert (src / "yieldpred" / "__init__.py").exists(), \
            f"{page.name}'s shim points at {src}, which has no yieldpred package"
        assert f"parents[{depth}]" in text, \
            f"{page.name} uses the wrong parents[] depth for its location"


def test_every_page_says_something():
    """A page that renders nothing at all is broken even if it doesn't raise."""
    for page in PAGES:
        app = AppTest.from_file(str(page), default_timeout=120).run()
        rendered = len(app.markdown) + len(app.warning) + len(app.title)
        assert rendered, f"{page.name} rendered no text"


def test_every_page_carries_the_disclaimer():
    """A new page must not ship without the legal footer.

    Checked statically rather than by rendering, because a page that stops early
    for missing data never reaches its footer - and the notice still has to be in
    the source. The text itself lives in one module so five pages can't drift.
    """
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        assert "from yieldpred.disclaimer import" in text, \
            f"{page.name} does not import the disclaimer"
        assert "st.caption(FOOTER)" in text, \
            f"{page.name} does not render the disclaimer footer"


def test_the_disclaimer_says_the_three_things_it_has_to():
    """Not advice, no endorsement, and accuracy in numbers."""
    from yieldpred.disclaimer import DATA_SOURCES, FOOTER, LIMITATIONS, README_BLOCK

    # Case-insensitive: the phrase starts a sentence in some of these.
    assert "advice" in FOOTER.lower()
    assert "not endorsed or certified" in FOOTER.lower()
    assert "bu/acre" in FOOTER                      # a number, not just a warning

    for agency in ("NASS", "NASA POWER", "USGS", "Census"):
        assert agency in DATA_SOURCES, f"{agency} missing from the source notice"
    assert "not endorsed or certified" in DATA_SOURCES.lower()

    for claim in ("bu/acre", "Moran", "retrospective", "2007", "2018"):
        assert claim in LIMITATIONS, f"limitations should mention {claim}"

    # The spatial-holdout leak is a known optimism in a headline number, so the
    # limitations panel has to carry it. Both figures, not just the caveat: a
    # reader who sees 0.781 quoted elsewhere needs the corrected one here.
    assert "0.023" in LIMITATIONS and "0.758" in LIMITATIONS, \
        "limitations must state the Iowa leak correction in numbers"

    assert "not endorsed or certified" in README_BLOCK.lower()


def test_every_page_wears_the_suite_chrome():
    """One call sets up every page, so six pages cannot drift apart.

    `page_setup()` is the only place that calls `st.set_page_config`, loads the
    logo, injects the suite CSS and draws the menu; `page_footer()` closes the
    page with the green band. A page that spelled any of that out itself would
    look right today and diverge at the next change, so this checks the calls
    rather than the appearance.
    """
    from yieldpred.brand import LOGO, NAV

    assert LOGO.exists(), f"missing {LOGO}"
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        assert "page_setup(" in text, f"{page.name} does not call page_setup()"
        assert "page_footer()" in text, f"{page.name} does not call page_footer()"
        assert "st.set_page_config" not in text, \
            f"{page.name} configures itself instead of going through page_setup()"


def test_the_menu_lists_exactly_the_pages_that_exist():
    """Every page is reachable from the menu, and every menu item is a page."""
    from yieldpred.brand import NAV

    listed = {(APP / path).resolve() for path, _, _, _ in NAV}
    assert listed == {p.resolve() for p in PAGES}, \
        "brand.NAV and app/pages/ disagree about which pages there are"


def test_every_menu_url_matches_the_url_streamlit_will_serve():
    """The pills are plain anchors, so a wrong href is a dead link.

    Streamlit serves a page at its filename with the sort prefix and the
    extension stripped: `pages/3_Model_and_Validation.py` becomes
    `Model_and_Validation`. NAV writes those out rather than deriving them,
    because the highlight and the navigation both depend on them and a wrong
    one fails silently. This derives them and checks.
    """
    import re

    from yieldpred.brand import NAV

    for path, label, _icon, url in NAV:
        if path == "streamlit_app.py":
            assert url == "", "the home page is served at the app root"
            continue
        expected = re.sub(r"^\d+_", "", Path(path).stem)
        assert url == expected, f"{label}: NAV says {url!r}, Streamlit serves {expected!r}"


def test_the_reading_order_numbers_every_page_but_home():
    """The pills are numbered as a suggested reading order, not as a wizard."""
    from yieldpred.brand import NAV, STEPS

    assert "Home" not in STEPS, "Home is the cover, not a step"
    assert sorted(STEPS.values()) == list(range(1, len(NAV))), \
        "the step numbers should run 1..n with no gaps or repeats"
    assert STEPS["How It Works"] == 1, \
        "the explanation comes first; it used to be buried at position four"


def test_the_menu_is_in_the_body_not_a_sidebar():
    """The left rail is gone; the green header only spans the window without it.

    Streamlit lays its header out beside the sidebar rather than above it
    (measured: header x=300 width=1100, sidebar x=0 width=300), so a visible
    sidebar cuts the green bar short - the look and the layout are one problem.
    A later change that reinstates a sidebar would regress the header silently,
    and this is the cheap guard against that.
    """
    from yieldpred.brand import SUITE_CSS

    assert '[data-testid="stSidebar"]' in SUITE_CSS and "display: none" in SUITE_CSS
    brand = (Path(__file__).resolve().parents[1] / "src" / "yieldpred" / "brand.py")
    assert 'initial_sidebar_state="collapsed"' in brand.read_text(encoding="utf-8"), \
        "the logo renders into the sidebar unless the sidebar starts collapsed"


def test_suite_css_matches_the_master_copy():
    """app/assets/suite.css and suite-apps.json are the farm app suite's shared files; the masters live with Herd Planner."""
    root = Path(__file__).resolve().parents[1]
    master = root.parent / "herd-planner" / "brand" / "suite.css"
    if not master.exists():
        pytest.skip("Herd Planner isn't checked out next to this project")
    for name in ("suite.css", "suite-apps.json"):
        copy, original = root / "app" / "assets" / name, master.parent / name
        assert copy.read_bytes().replace(b"\r\n", b"\n") == original.read_bytes().replace(b"\r\n", b"\n"), \
            f"{name} drifted: copy herd-planner/brand/{name} here again"


def test_the_farm_apps_list_names_every_app_and_marks_this_one():
    """The cross-app list, which now rides in the footer band rather than the
    sidebar. Both renderings read the same JSON, so both are checked."""
    from yieldpred.brand import _suite_menu_html, suite_menu_markdown

    from yieldpred.brand import APP_NAME

    for text in (suite_menu_markdown(), _suite_menu_html()):
        for name in ("Herd Planner", APP_NAME, "Farm Equipment Planner"):
            assert name in text
        assert "you're here" in text

    assert f"{APP_NAME} (you're here)" in suite_menu_markdown()


# --------------------------------------------------- the suite chrome, in detail

def test_the_stylesheet_has_no_angle_brackets():
    """A `<` inside SUITE_CSS silently destroys the entire stylesheet.

    `st.html` runs its argument through an HTML sanitizer, and the sanitizer
    reads the text inside a style element as markup. A CSS comment that
    mentioned a class name written with the key in angle brackets looked like
    an opening tag, and the whole style block was dropped - no exception, no
    warning in the server log, just an app rendered with no styling at all. It
    took two debugging passes precisely because the CSS itself was valid.

    So: no angle brackets in the stylesheet, not even in a comment. Child
    combinators would have to be written some other way if one is ever needed.
    """
    from yieldpred.brand import SUITE_CSS

    inner = SUITE_CSS.replace("<style>", "").replace("</style>", "")
    assert "<" not in inner and ">" not in inner, (
        "an angle bracket inside SUITE_CSS makes st.html's sanitizer discard "
        "the whole stylesheet; found: "
        + repr([line for line in inner.splitlines() if "<" in line or ">" in line]))


def _relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance, so contrast is computed and not eyeballed."""
    channels = []
    for offset in (1, 3, 5):
        value = int(hex_color[offset:offset + 2], 16) / 255
        channels.append(value / 12.92 if value <= 0.03928
                        else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(one: str, two: str) -> float:
    a, b = sorted((_relative_luminance(one), _relative_luminance(two)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def test_the_header_and_menu_colours_are_readable():
    """The menu sits on deep green, so its text has to be checked, not assumed.

    White on the green passes comfortably. The active item is gold, and the
    thing to protect against is someone "tidying" it to white text: white on
    this gold is about 2.2:1, which is unreadable, and it is an easy mistake
    because white text is what every other item in the bar uses.
    """
    from yieldpred.brand import DEEP_GREEN, GOLD

    assert _contrast("#ffffff", DEEP_GREEN) >= 7.0, "white on the green bar"
    assert _contrast(DEEP_GREEN, GOLD) >= 4.5, "active item: deep green on gold"
    assert _contrast("#ffffff", GOLD) < 3.0, (
        "if this ever passes, the gold changed - recheck the active item's text")


def test_the_header_bar_names_the_app():
    """The app's name is in the green bar, and it is not the old wrong one."""
    from yieldpred import brand

    assert brand.APP_NAME == "Yield Predictor"
    for stale in ("Nebraska Corn Yield Predictor", "Corn Yield Predictor"):
        assert stale != brand.APP_NAME
    assert "suite-headerbar" in brand.SUITE_CSS, "no styling for the header name"


def test_the_header_bar_links_back_to_the_company_site_first():
    """Top left, before the app's name: the way home every app in the suite has."""
    from yieldpred import brand

    company = brand.load_company()
    html = brand.header_bar_html()
    assert f'href="{company["url"]}" target="_top">{company["name"]}</a>' in html
    assert html.index('class="company"') < html.index('class="name"')
    assert "pointer-events: auto" in brand.SUITE_CSS, "the bar ignores clicks; the link must take them back"


def test_the_company_sits_above_the_app_name():
    """Suite version 6: "Cornerpost Logic ›" stacked over the app's name, one picture (the app's logo),
    as in every app; no company icon beside it any more."""
    from yieldpred import brand

    html = brand.header_bar_html()
    titles = html[html.index('class="titles"'):]
    assert titles.index('class="company"') < titles.index('class="name"')
    assert "flex-direction: column" in brand.SUITE_CSS
    assert "suite-company-icon" not in brand.SUITE_CSS and 'class="sep"' not in html


def test_no_page_still_carries_the_old_app_name():
    """The rename has to be complete, or the tab and the bar disagree."""
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        assert "Nebraska Corn Yield Predictor" not in text, \
            f"{page.name} still uses the old app name"


def test_the_footer_rules_come_from_the_shared_suite_css():
    """This app inlines the suite's footer rules; it cannot link them.

    Streamlit's `st.html` sanitizer strips a stylesheet link element, so unlike
    the other two apps and the website this one can't load suite.css. It reads
    the marked section instead, which keeps one source of truth. If the markers
    are ever dropped from suite.css - say by copying an older master over it -
    the footer silently loses its layout, so this checks them.
    """
    from yieldpred.brand import (FOOTER_MARK_END, FOOTER_MARK_START, SUITE_CSS,
                                 SUITE_CSS_FILE, shared_footer_css)

    text = SUITE_CSS_FILE.read_text(encoding="utf-8")
    assert FOOTER_MARK_START in text and FOOTER_MARK_END in text, \
        "suite.css lost the markers that delimit the shared footer rules"

    section = shared_footer_css()

    # The slice has to be valid CSS standing alone. The markers were once the
    # opening line of a long comment, so the extract began mid-comment and
    # carried an orphaned close; the browser read that as a broken selector and
    # its error recovery ate the next rule - the one setting the background. The
    # footer rendered with no green while everything after it applied normally.
    assert section.count("/*") == section.count("*/"), \
        "the extracted footer section begins or ends inside a comment"
    assert section.lstrip().startswith("."), \
        "the extracted footer section should start at a selector"
    for rule in (".footer-inner", ".footer-brand", ".footer-col", ".footer-bottom"):
        assert rule in section, f"the shared footer section has no {rule} rule"
    assert section in SUITE_CSS, "the shared rules were not inlined into the page CSS"

    # The section is injected through st.html, whose sanitizer discards a whole
    # style block over a single angle bracket. suite.css is edited by people
    # working on three other codebases, so this is checked here too.
    assert "<" not in section and ">" not in section, \
        "an angle bracket in suite.css's footer section would blank this app's styling"

    # Every var() needs a fallback. The other apps inject suite.css whole, so
    # :root is always there for them; this app injects the marked section alone,
    # and a bare var(--suite-deep-green) resolved to nothing - the footer lost
    # its green and rendered white text on paper.
    import re

    # Comments are stripped first: the section's own comment explains the rule
    # by quoting a bare var(), and a check that reads comments as code flags it.
    rules = re.sub(r"/\*.*?\*/", "", "/*" + section, flags=re.S)
    bare = [v for v in re.findall(r"var\(--[a-z-]+[^)]*\)", rules) if "," not in v]
    assert not bare, f"var() without a fallback in the shared footer section: {bare}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
@pytest.mark.parametrize("state", ["ne", "ia"])
def test_every_page_runs_for_soybeans(page, state):
    """The crop switch (?crop=soybeans): every page, both states, no exception."""
    app = AppTest.from_file(str(page), default_timeout=120)
    app.query_params["crop"] = "soybeans"
    app.query_params["state"] = state
    app.run()
    assert not app.exception, f"{page.name} ({state}, soybeans): {[e.value for e in app.exception]}"


def test_links_keep_the_crop_and_the_state():
    """Following a menu pill, the state switch or a footer link must not drop back to Nebraska corn."""
    from yieldpred import brand

    assert brand.view_query() == ""                                  # Nebraska corn keeps plain URLs
    assert brand.view_query("ia") == "?state=ia"
    assert brand.view_query("ne", "soybeans") == "?crop=soybeans"
    assert brand.view_query("ia", "soybeans") == "?state=ia&crop=soybeans"
    assert "Iowa Soybeans" in brand.header_bar_html("ia", "soybeans")
    assert "Nebraska Corn" in brand.header_bar_html()
    assert brand.for_crop(brand.QUESTIONS["Maps"], "soybeans").startswith("Where do soybeans do well")
    assert brand.for_crop("Predicting county corn yields", "soybeans") == "Predicting county soybean yields"


def test_the_switches_read_state_then_crop():
    """ "Nebraska Corn", the way a farmer says it: the state switch comes before the crop switch."""
    import streamlit as st

    from yieldpred import brand

    html = []
    real = st.html
    st.html = html.append
    try:
        brand.nav_bar("County Explorer")
    finally:
        st.html = real
    page = html[0]
    assert page.index(">Nebraska<") < page.index(">Corn<")


def test_streamlit_never_draws_its_own_page_list():
    """Streamlit's default page list lives in a left sidebar; drawn before brand.py's CSS hides it, it
    flashed the old layout on every visit."""
    import tomllib
    from pathlib import Path

    config = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text())
    assert config["client"]["showSidebarNavigation"] is False


def _menu_html(current="Maps", state="ne", crop="corn"):
    import streamlit as st

    from yieldpred import brand

    html, real = [], st.html
    st.html = html.append
    try:
        brand.nav_bar(current, state, crop)
    finally:
        st.html = real
    return html[0]


def test_one_step_bar_names_and_numbers_every_page():
    """The menu is the only step bar: every page gets a card with the menu's own number and name,
    and a few words on what it answers (Home's separate Start Here cards numbered only three)."""
    from yieldpred import brand

    page = _menu_html()
    assert page.count('class="step') == len(brand.STEPS)  # Home is in the green header
    for label, step in brand.STEPS.items():
        assert f'<span class="n">{step}</span><span class="l">{label}</span>' in page
    assert set(brand.NAV_BLURBS) == set(brand.STEPS)
    assert 'class="step active" href="Maps" aria-current="page"' in page
    assert not hasattr(brand, "start_here")


def test_the_menu_words_follow_the_crop():
    page = _menu_html(crop="soybeans")
    assert "Where soybeans do well" in page and "Where corn does well" not in page


def test_home_is_in_the_green_header():
    """Home has no step number, so as a card it broke the numbered order on phones (1, 3, 5 on the
    right). It sits in the green header instead, and the app's name there links home too."""
    from yieldpred import brand

    page = _menu_html("Maps")
    assert ">Home<" not in page
    head = brand.header_bar_html("ne", "soybeans", current="Maps")
    assert 'class="home" href="./?crop=soybeans"' in head and 'class="name" href="./?crop=soybeans"' in head
    assert 'class="home active"' in brand.header_bar_html(current="Home")


def test_home_says_what_honest_means_and_links_to_the_proof():
    """No hover (phones never show one): a plain line under the question, and the words
    "how well that works" link to Model & Validation, keeping the state and crop."""
    import streamlit as st

    from yieldpred import brand

    shown, real = [], st.html
    st.html = shown.append
    real_state, real_crop = brand.current_state, brand.current_crop
    brand.current_state, brand.current_crop = (lambda: "ia"), (lambda: "soybeans")
    try:
        brand.page_heading("Home", show_title=False)
    finally:
        st.html, brand.current_state, brand.current_crop = real, real_state, real_crop
    page = shown[0]
    assert '<a href="Model_and_Validation?state=ia&crop=soybeans" target="_self">how well that works</a>' in page
    assert brand.HONEST_NOTE in page and "soybean" in page


# --- The logo, redrawn as a line symbol (October 2026) --------------------------------------------
# One family across the farm app suite: a rounded-square tile in the app's colour, a cream line
# symbol, one gold piece. The masters and the rules are in herd-planner/brand (README.md there).

def _drawing(path):
    """The drawing in an SVG file. Some tools add a note about where a file came from (a <metadata>
    block) when they save it; that note isn't part of the picture, so two copies of one drawing still
    compare as equal."""
    import re

    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    text = re.sub(r"<metadata>.*?</metadata>", "", text, flags=re.S)
    return text.replace(' xmlns:c2pa="http://c2pa.org/manifest"', "")


def _png_shape(path):
    """(width, height, has see-through corners), read from the file's first block: no picture library needed."""
    head = path.read_bytes()[:26]
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR", path.name
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"), head[25] == 6


def test_the_logo_is_the_corn_ear_on_a_rounded_brown_tile():
    """Holds without Herd Planner beside the project."""
    import re

    assets = Path(__file__).resolve().parents[1] / "app" / "assets"
    svg = _drawing(assets / "logo.svg")
    assert '<rect x="2" y="2" width="96" height="96" rx="22" fill="#4a3520"/>' in svg
    assert set(re.findall(r"#[0-9a-f]{6}", svg)) == {"#4a3520", "#f4efe3", "#d9a441"}   # tile, cream line, gold ear
    assert _png_shape(assets / "logo.png") == (256, 256, True)       # what st.logo and the footer show
    assert _png_shape(assets / "favicon.png") == (64, 64, True)      # the browser tab


def test_the_browser_tab_uses_the_small_cut():
    """The regular cut's line is under one pixel wide at 16 px; favicon.png is drawn from the heavier small cut."""
    from yieldpred import brand

    assert Path(brand.PAGE_ICON).name == "favicon.png" and Path(brand.PAGE_ICON).exists()
    assert brand.LOGO.name == "logo.png" and brand.LOGO.exists()
    source = Path(brand.__file__).read_text(encoding="utf-8")
    assert "page_icon=PAGE_ICON" in source and "st.logo(str(LOGO)" in source


def test_the_logo_matches_the_master_copy():
    root = Path(__file__).resolve().parents[1]
    master = root.parent / "herd-planner" / "brand" / "corn-yield-predictor-logo.svg"
    if not (master.parent / "corn-yield-predictor-logo-small.svg").exists():
        pytest.skip("Herd Planner with the redrawn logos isn't checked out next to this project")
    assert _drawing(root / "app" / "assets" / "logo.svg") == _drawing(master), \
        "logo.svg drifted: copy herd-planner/brand/corn-yield-predictor-logo.svg here again"


def test_the_header_logo_is_the_suites_size():
    """Scott, 2026-10-07, once the new logos were live: "Both logos seem a bit small" (this app's and the
    Equipment Planner's). Streamlit's largest logo is 32 px; the other apps in the suite draw theirs at 40,
    so the suite CSS makes this one 40 and moves the name right by the same 8 px."""
    from yieldpred import brand

    assert brand.LOGO_PX == 40
    assert '[data-testid="stHeaderLogo"] { height: 40px !important; width: 40px !important;' in brand.SUITE_CSS
    assert "left: 68px;" in brand.SUITE_CSS          # 16 px margin + the 40 px logo + a 12 px gap
    assert brand.LOGO_PX < brand.HEADER_HEIGHT_PX    # it has to fit in Streamlit's 60 px header
