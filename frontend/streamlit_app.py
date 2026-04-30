import streamlit as st

st.set_page_config(layout="wide")

dashboard_page = st.Page(
    "pages/dashboard.py", 
    title="Dashboard", 
    default=True
)

prediction_page = st.Page(
    "pages/prediction.py", 
    title="Prédiction", 
)

comparaison_page = st.Page(
    "pages/comparaison.py", 
    title="Comparaison des modèles", 
)

analyse_page = st.Page(
    "pages/analyse.py", 
    title="Analyse", 
)

pg = st.navigation({
    "Pages": [dashboard_page, prediction_page, comparaison_page, analyse_page]
})
pg.run()

