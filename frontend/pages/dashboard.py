import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import requests

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

def get_risk_scores(current_customers_df):
    
    risk_scores = []
    url = 'http://127.0.0.1:8000/predict'
    
    # barre de progression car l'appel en boucle peut prendre du temps
    progress_bar = st.progress(0, text="Analyse du risque via l'IA...")
    
    # Limité à 50 pour le test de performance
    total = len(current_customers_df.head(50)) 
    
    for i, (index, row) in enumerate(current_customers_df.head(50).iterrows()):
        payload = {
            "gender": row['gender'],
            "age": row['age'],
            "country": row['country'],
            "city": row['city'],
            "customer_segment": row['customer_segment'],
            "tenure_months": row['tenure_months'],
            "signup_channel": row['signup_channel'],
            "contract_type": row['contract_type'],
            "monthly_logins": row['monthly_logins'],
            "weekly_active_days": row['weekly_active_days'],
            "avg_session_time": row['avg_session_time'],
            "features_used": row['features_used'],
            "usage_growth_rate": row['usage_growth_rate'],
            "last_login_days_ago": row['last_login_days_ago'],
            "monthly_fee": row['monthly_fee'],
            "total_revenue": row['total_revenue'],
            "payment_method": row['payment_method'],
            "payment_failures": row['payment_failures'],
            "discount_applied": row['discount_applied'],
            "price_increase_last_3m": row['price_increase_last_3m'],
            "support_tickets": row['support_tickets'],
            "avg_resolution_time": row['avg_resolution_time'],
            "complaint_type": row['complaint_type'] if pd.notna(row['complaint_type']) else "None",
            "csat_score": float(row['csat_score']),
            "escalations": row['escalations'],
            "email_open_rate": row['email_open_rate'],
            "marketing_click_rate": row['marketing_click_rate'],
            "nps_score": row['nps_score'],
            "survey_response": row['survey_response'],
            "referral_count": row['referral_count']
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                prob = response.json().get("churn_probability", 0)
                risk_scores.append(prob)
            else:
                risk_scores.append(0)
        except:
            risk_scores.append(0)
            
        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty() 
    return risk_scores

st.header(" Clients à haut risque")

current_customers = df[df['churn'] == 0].copy().head(50)

current_customers['risk_score'] = get_risk_scores(current_customers)

top_risk_df = current_customers.sort_values(by='risk_score', ascending=False).head(10)

display_columns = [
    'customer_id', 'risk_score', 'customer_segment', 'tenure_months', 
    'csat_score', 'support_tickets', 'monthly_fee'
]

st.dataframe(
    top_risk_df[display_columns].style.format({'risk_score': '{:.1%}'})
    .background_gradient(subset=['risk_score'], cmap='OrRd')
    .background_gradient(subset=['csat_score'], cmap='RdYlGn', vmin=1, vmax=5),
    width="stretch",
    hide_index=True
)