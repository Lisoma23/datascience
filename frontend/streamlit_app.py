import streamlit as st


st.set_page_config(layout="wide")

def dashboard():
    st.title("Dashboard")

def chart():
    st.title("Chart")

pages = [
    st.Page(dashboard, title="Dashboard"),
    st.Page(chart, title="Chart"),
]

pg = st.navigation(pages)
pg.run()