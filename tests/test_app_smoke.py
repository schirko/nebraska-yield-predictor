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
    """This is what lets nav_bar() swallow StreamlitPageNotFoundError safely.

    `st.page_link` resolves against the entrypoint file, so under
    `AppTest.from_file(one_page)` the other five pages aren't registered and the
    menu has to skip them or take every page test down. Swallowing that error at
    runtime would normally risk hiding a typo in NAV - it doesn't, because this
    test pins NAV and app/pages/ to the same set of files. Static check catches
    the typo; the runtime catch only ever fires for a page that genuinely isn't
    part of the app being run.
    """
    from yieldpred.brand import NAV

    listed = {(APP / path).resolve() for path, _, _ in NAV}
    assert listed == {p.resolve() for p in PAGES}, \
        "brand.NAV and app/pages/ disagree about which pages there are"


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

    for text in (suite_menu_markdown(), _suite_menu_html()):
        for name in ("Herd Planner", "Corn Yield Predictor", "Farm Equipment Planner"):
            assert name in text
        assert "you're here" in text

    text = suite_menu_markdown()
    for name in ("Herd Planner", "Corn Yield Predictor", "Farm Equipment Planner"):
        assert name in text
    assert "Corn Yield Predictor (you're here)" in text
