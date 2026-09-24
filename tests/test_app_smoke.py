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


def test_every_page_says_something():
    """A page that renders nothing at all is broken even if it doesn't raise."""
    for page in PAGES:
        app = AppTest.from_file(str(page), default_timeout=120).run()
        rendered = len(app.markdown) + len(app.warning) + len(app.title)
        assert rendered, f"{page.name} rendered no text"
