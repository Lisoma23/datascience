import streamlit as st

st.set_page_config(layout="wide")

dashboard_page = st.Page(
    "pages/dashboard.py", 
    title="Dashboard", 
    default=True
)

chart_page = st.Page(
    "pages/chart.py", 
    title="Chart", 
)

pg = st.navigation({
    "Pages": [dashboard_page, chart_page]
})
pg.run()

