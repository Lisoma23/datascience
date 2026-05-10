from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
url_api = 'http://127.0.0.1:8000/predict'


st.title("Simulateur de Scénarios de Churn")
st.subheader("Modifiez les paramètres pour tester l'impact sur la fidélité client")

# --- 1. PROFIL PAR DÉFAUT (Valeurs moyennes types) ---
default_profile = {
    "gender": "Male",
    "age": 40,
    "country": "Germany",
    "city": "Berlin",
    "customer_segment": "Individual",
    "tenure_months": 24,
    "signup_channel": "Web",
    "contract_type": "Monthly",
    "monthly_logins": 12,
    "weekly_active_days": 3,
    "avg_session_time": 15.0,
    "features_used": 3,
    "usage_growth_rate": 0.05,
    "last_login_days_ago": 4,
    "monthly_fee": 50.0,
    "total_revenue": 1200.0,
    "payment_method": "Card",
    "payment_failures": 0,
    "discount_applied": "No",
    "price_increase_last_3m": "No",
    "support_tickets": 1,
    "avg_resolution_time": 5.0,
    "complaint_type": None,
    "csat_score": 4.0,
    "escalations": 0,
    "email_open_rate": 0.4,
    "marketing_click_rate": 0.1,
    "nps_score": 30,
    "survey_response": "Neutral",
    "referral_count": 1
}

# --- SIDE BAR ---
st.sidebar.header("Ajuster le Scénario")

simulated_profile = default_profile.copy()

with st.sidebar:
    st.markdown("### Offre & Tarification")
    simulated_profile["monthly_fee"] = st.slider(
        "Frais Mensuels ($)", 10.0, 250.0, float(default_profile["monthly_fee"]), step=1.0
    )
    simulated_profile["discount_applied"] = st.radio(
        "Accorder une remise ?", ["No", "Yes"],
        index=0 if default_profile["discount_applied"] == "No" else 1
    )

    st.divider()
    st.markdown("### Engagement & Usage")
    simulated_profile["usage_growth_rate"] = st.slider(
        "Taux de croissance usage", -1.0, 1.0, float(default_profile["usage_growth_rate"]), step=0.05
    )
    simulated_profile["referral_count"] = st.number_input(
        "Nombre de parrainages", 0, 50, int(default_profile["referral_count"])
    )

    st.divider()
    st.markdown("### Support & Satisfaction")
    # Correction de l'erreur types : min/max/value/step sont TOUS en float
    simulated_profile["csat_score"] = st.slider(
        "Score de satisfaction (CSAT)", 1.0, 5.0, float(default_profile["csat_score"]), step=0.5
    )
    simulated_profile["support_tickets"] = st.slider(
        "Nombre de tickets support", 0, 20, int(default_profile["support_tickets"]), step=1
    )

# --- FONCTION APPEL API ---
def get_churn_prediction(profile_data):
    try:
        response = requests.post(url_api, json=profile_data)
        if response.status_code == 200:
            return response.json().get("churn_probability", 0.0)
        st.warning(f"Erreur API ({response.status_code})")
        return None
    except requests.exceptions.ConnectionError:
        st.warning("API indisponible — lancez le serveur avec `make api`")
        return None

# --- CALCULS ET COMPARAISON ---
prob_initiale = get_churn_prediction(default_profile)
prob_simulee = get_churn_prediction(simulated_profile)

if prob_initiale is None or prob_simulee is None:
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risque Initial", f"{prob_initiale:.1%}")

with col2:
    diff = prob_simulee - prob_initiale
    st.metric("Nouveau Risque", f"{prob_simulee:.1%}", delta=f"{diff:.1%}", delta_color="inverse")

with col3:
    impact = "Positif" if diff < 0 else "Négatif" if diff > 0 else "Nul"
    st.metric("Impact du Scénario", impact)

# --- CHART ---
st.divider()
col_graph, col_txt = st.columns([2, 1])

with col_graph:
    chart_data = pd.DataFrame({
        "Scénario": ["Initial", "Simulé"],
        "Probabilité de Churn (%)": [prob_initiale * 100, prob_simulee * 100]
    })

    fig = px.bar(
        chart_data, x="Scénario", y="Probabilité de Churn (%)",
        color="Scénario",
        color_discrete_map={"Initial": "#4a4f5c", "Simulé": "#ff4b4b"},
        text_auto='.1f'
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")

with col_txt:
    st.markdown("### Analyse")
    if diff < -0.05:
        st.success("""
        **Favorable** : Les modifications apportées réduisent significativement le risque de départ.
        """)
    elif diff > 0.05:
        st.error("""
        **Risqué** : Ce scénario augmente la probabilité que le client nous quitte.
        """)
    else:
        st.info("L'impact de ces changements n'est pas impactant sur la prédiction du modèle.")

st.divider()
st.expander("Voir les données brutes envoyées").json(simulated_profile)
