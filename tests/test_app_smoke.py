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

    assert "not endorsed or certified" in README_BLOCK.lower()
