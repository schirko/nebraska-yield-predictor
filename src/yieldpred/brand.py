"""The app's logo: an ear of corn on a soil-brown circle.

Same family as Herd Planner's cow and Farm Equipment Planner's tractor: a
colored circle, a cream drawing, a wheat-gold accent. The master SVG is
app/assets/logo.svg; logo.png is rendered from it because Streamlit's tab icon
and st.logo are most reliable with a PNG.

Every page calls show_logo() right after st.set_page_config(), and passes
PAGE_ICON as its page_icon, so the browser tab and the top-left corner carry the
logo on every page.
"""

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "app" / "assets"
LOGO = ASSETS / "logo.png"
PAGE_ICON = str(LOGO)


def show_logo():
    """Put the logo in the top-left corner (above the page list in the sidebar)."""
    import streamlit as st  # imported here so the pipeline can import yieldpred without Streamlit

    st.logo(str(LOGO), size="large", icon_image=str(LOGO))
