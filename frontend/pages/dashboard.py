import streamlit as st
import pandas as pd
import plotly.express as px

# --- FAKE DATA---
data = [
    {"Segment": "Premium", "Contract": "Two year", "Churn": "No", "CSAT": 9, "Revenue": 150},
    {"Segment": "Basic", "Contract": "Month-to-month", "Churn": "Yes", "CSAT": 3, "Revenue": 45},
    {"Segment": "Medium", "Contract": "One year", "Churn": "No", "CSAT": 7, "Revenue": 85},
    {"Segment": "Basic", "Contract": "Month-to-month", "Churn": "Yes", "CSAT": 2, "Revenue": 40},
    {"Segment": "Premium", "Contract": "One year", "Churn": "No", "CSAT": 8, "Revenue": 140},
    {"Segment": "Medium", "Contract": "Month-to-month", "Churn": "No", "CSAT": 6, "Revenue": 75},
    {"Segment": "Basic", "Contract": "Two year", "Churn": "No", "CSAT": 9, "Revenue": 50},
    {"Segment": "Basic", "Contract": "Month-to-month", "Churn": "Yes", "CSAT": 4, "Revenue": 42},
    {"Segment": "Medium", "Contract": "One year", "Churn": "No", "CSAT": 8, "Revenue": 90},
    {"Segment": "Premium", "Contract": "Two year", "Churn": "No", "CSAT": 10, "Revenue": 160},
]
df = pd.DataFrame(data)

total_clients = 10000 
churn_rate = 10.2 
revenu_total = 53456
revenu_risque = int(revenu_total * (churn_rate / 100))


st.title("KPIs")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Nombre total de clients", value=f"{total_clients:,}", delta="-1.2%")
with col2:
    st.metric(label="Taux de churn global", value=f"{churn_rate}%", delta="0.3%")
with col3:
    st.metric(label="Revenu total", value=f"{revenu_total:,} $", delta="333 $")
with col4:
    st.metric(label="Revenu à risque", value=f"{revenu_risque:,} $", delta="-32 $")

st.markdown("---")

st.title("Graphiques")

colors = {'No': '#4a4f5c', 'Yes': '#ff4b4b'}

row1_col1, row1_col2 = st.columns(2)

# Distribution du Churn
with row1_col1:
    fig_pie = px.pie(
        df, names='Churn', 
        title="Répartition des clients (Churn vs Non-Churn)",
        color='Churn',
        color_discrete_map=colors,
        hole=0.5
    )
    st.plotly_chart(fig_pie, width='stretch')

# Churn par Segment
with row1_col2:
    fig_seg = px.bar(
        df, x="Segment", color="Churn", 
        title="Churn par Segment Client",
        color_discrete_map=colors,
    )
    st.plotly_chart(fig_seg, width='stretch')

row2_col1, row2_col2 = st.columns(2)

# Churn par type de contrat
with row2_col1:
    fig_contract = px.bar(
        df, x="Contract", color="Churn", 
        title="Churn par Type de Contrat",
        color_discrete_map=colors
    )
    st.plotly_chart(fig_contract, width='stretch')

# Distribution CSAT
with row2_col2:
    fig_csat = px.box(
        df, x="Churn", y="CSAT", color="Churn",
        title="Satisfaction Client (CSAT) par Statut",
        color_discrete_map=colors
    )
    st.plotly_chart(fig_csat, width='stretch')