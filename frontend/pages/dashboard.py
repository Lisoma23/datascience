import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

_CSV_PATH = Path(__file__).resolve().parent.parent.parent
csv_path = _CSV_PATH / "customer_churn.csv"

df = pd.read_csv(csv_path)

# --- CALCUL DES KPIs ---
total_clients = len(df)
churn_rate = round(df["churn"].mean() * 100, 2)
revenu_total = int(df["monthly_fee"].sum())
revenu_risque = int(df[df["churn"] == 1]["monthly_fee"].sum())

# --- AFFICHAGE KPIs ---
st.title("KPIs")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clients", f"{total_clients:,}")
col2.metric("Taux de Churn", f"{churn_rate}%")
col3.metric("Revenu Mensuel Total", f"{revenu_total:,} $")
col4.metric("Revenu à Risque", f"{revenu_risque:,} $", delta_color="inverse")

st.divider()

# --- GRAPHIQUES ---
st.header(" Analyses Graphiques")

# On définit les couleurs pour la cohérence
colors = {'No': '#4a4f5c', 'Yes': '#ff4b4b'}

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig_pie = px.pie(
        df, names='churn', 
        title="Répartition Churn vs Non-Churn",
        color='churn',
        color_discrete_map=colors,
        hole=0.5
    )
    st.plotly_chart(fig_pie, width='stretch')

with row1_col2:
    # Tri pour avoir une meilleure lecture
    fig_seg = px.histogram(
        df, x="customer_segment", color="churn", 
        title="Churn par Segment Client",
        barmode="group",
        color_discrete_map=colors,
    )
    st.plotly_chart(fig_seg, width='stretch')

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    fig_contract = px.histogram(
        df, x="contract_type", color="churn", 
        title="Churn par Type de Contrat",
        barmode="group",
        color_discrete_map=colors
    )
    st.plotly_chart(fig_contract, width='stretch')

with row2_col2:
    fig_csat = px.box(
        df, x="churn", y="csat_score", color="churn",
        title="Score de Satisfaction (CSAT) par Statut",
        color_discrete_map=colors
    )
    st.plotly_chart(fig_csat, width='stretch')


# --- CLIENTS À RISQUE ---
st.divider()
st.header(" Clients à haut risque")

customers_at_risk = df[df['churn'] == 0].copy()

# Calcul d'un score de risque simple pour le tri
customers_at_risk['risk_score'] = (
    (5 - customers_at_risk['csat_score']) * 2 + 
    customers_at_risk['support_tickets'] + 
    customers_at_risk['escalations'] * 2
)

# Tri par score de risque
top_risk_df = customers_at_risk.sort_values(by='risk_score', ascending=False).head(10)

# Sélection des colonnes pour l'affichage
display_columns = [
    'customer_id', 'customer_segment', 'tenure_months', 
    'csat_score', 'support_tickets', 'escalations', 'monthly_fee'
]

st.dataframe(
    top_risk_df[display_columns].style.background_gradient(
        subset=['csat_score'], cmap='RdYlGn', vmin=1, 
        vmax=5
    ).background_gradient(
        subset=['support_tickets', 'escalations'], cmap='OrRd'
    ),
    width="stretch",
    hide_index=True
)