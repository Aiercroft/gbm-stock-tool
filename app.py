import streamlit as st

st.set_page_config(
    page_title="Finance Recorder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "savings" not in st.session_state:
    st.session_state.savings = {
        "geometric_brownian": []
    }

st.sidebar.title("Finance Recorder")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Model",
    ["Geometric Brownian Motion"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Statistical Models for Finance")

if page == "Geometric Brownian Motion":
    from pages.geometric_brownian import render_page
    render_page()
