import streamlit as st

st.set_page_config(
    page_title="Geometric Brownian Motion",
    page_icon="∿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "savings" not in st.session_state:
    st.session_state.savings = {
        "geometric_brownian": []
    }

from pages.geometric_brownian import render_page

render_page()
