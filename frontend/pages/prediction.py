import streamlit as st
import requests

st.title("Prédiction du Churn en Temps Réel")
st.subheader("Simulez un profil client pour évaluer son risque de départ")

with st.form("prediction_form"):
    
    st.markdown("### Informations Client & Contrat")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Démographie**")
        age = st.number_input("Âge", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Genre", ["Male", "Female"])
        country = st.selectbox("Pays", ["UK", "USA", "Canada", "Germany", "India", "Australia", "Bangladesh"])
        city = st.text_input("Ville", value="London")
        
    with col2:
        st.write("**Contrat**")
        tenure = st.number_input("Ancienneté (mois)", min_value=0, value=12)
        segment = st.selectbox("Segment Client", ["Individual", "SME", "Enterprise"])
        contract = st.selectbox("Type de contrat", ["Monthly", "Quarterly", "Yearly"])
        channel = st.selectbox("Canal d'acquisitation", ["Web", "Mobile", "Referral"])
        payment = st.selectbox("Méthode de paiement", ["PayPal", "Card", "Bank Transfer"])

    st.divider()

    st.markdown("### Usage & Engagement")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Statistiques d'utilisation**")
        logins = st.number_input("Connexions mensuelles", value=20)
        active_days = st.slider("Jours actifs par semaine", 0, 7, 4)
        session_time = st.number_input("Temps moyen de session (min)", value=15.0)
        last_login = st.number_input("Dernière connexion (jours)", value=2)
        growth_rate = st.slider("Taux de croissance de l'usage (%)", -100, 100, 0)
        
    with col4:
        st.write("**Marketing & Social**")
        email_rate = st.slider("Taux d'ouverture email (%)", 0, 100, 25)
        click_rate = st.slider("Taux de clic marketing (%)", 0, 100, 5)
        referrals = st.number_input("Nombre de parrainages", value=0)
        features = st.number_input("Nombre de fonctionnalités utilisées", min_value=0, value=3)

    st.divider()

    st.markdown("### Finance & Satisfaction")
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("**Données Financières**")
        monthly_fee = st.number_input("Frais mensuels ($)", value=30)
        total_rev = st.number_input("Revenu total généré ($)", value=500)
        discount = st.radio("Remise appliquée ?", ["No", "Yes"], horizontal=True)
        failures = st.number_input("Échecs de paiement", value=0)
        price_inc = st.selectbox("Augmentation de prix (3m) ?", ["No", "Yes"])     

    with col6:
        st.write("**Satisfaction (KPIs Clés)**")
        csat = st.slider("Score CSAT (1-5)", 1, 5, 4)
        nps = st.slider("Score NPS", -100, 100, 20)
        tickets = st.number_input("Tickets de support ouverts", value=1)
        res_time = st.number_input("Temps de résolution moy. (h)", value=2)
        escalations = st.number_input("Nombre d'escalades", value=0)
        complaint = st.selectbox("Type de réclamation", ["None", "Service", "Billing", "Technical"])
        survey = st.selectbox("Dernier sondage", ["Satisfied", "Neutral", "Unsatisfied"])

    submit_button = st.form_submit_button("Calculer la probabilité de Churn")


if submit_button:
    data_to_send = {
        "gender": gender,
        "age": age,
        "country": country,
        "city": city,
        "customer_segment": segment,
        "tenure_months": tenure,
        "signup_channel": channel,
        "contract_type": contract,
        "monthly_logins": logins,
        "weekly_active_days": active_days,
        "avg_session_time": session_time,
        "features_used": features,
        "usage_growth_rate": growth_rate / 100.0,
        "last_login_days_ago": last_login,
        "monthly_fee": monthly_fee,
        "total_revenue": total_rev,
        "payment_method": payment,
        "payment_failures": failures,
        "discount_applied": discount,
        "price_increase_last_3m": price_inc,
        "support_tickets": tickets,
        "avg_resolution_time": res_time,
        "complaint_type": complaint if complaint != "None" else None,
        "csat_score": float(csat),
        "escalations": escalations,
        "email_open_rate": email_rate / 100.0,
        "marketing_click_rate": click_rate / 100.0,
        "nps_score": nps,
        "survey_response": survey,
        "referral_count": referrals
    }

    url = 'http://127.0.0.1:8000/predict'

    try:
        with st.spinner("Appel à l'API de prédiction..."):
            response = requests.post(url, json=data_to_send)
            if response.status_code == 200:
                prediction_data = response.json()
                
                prob = prediction_data["churn_probability"] * 100
                is_churn = prediction_data["churn_prediction"] == 1
                
                statut = "CHURN" if is_churn else "NON-CHURN"
                couleur = "#FF4B4B" if is_churn else "#28A745"

                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.markdown("### Prédiction")
                    st.markdown(
                        f"<div style='padding: 10px; border-radius:10px; background-color:{couleur}; "
                        f"color:white; text-align:center; font-weight:bold;'>{statut}</div>", 
                        unsafe_allow_html=True
                    )

                with col_res2:
                    st.markdown(f"### Probabilité : {prob:.1f}%")
                    st.markdown(f"""
                                <div style="width: 100%; background-color: #f0f2f6; border-radius: 10px; margin-top:15px;">
                                <div style="width: {prob}%; background-color: {couleur}; 
                                height: 15px; border-radius: 10px; transition: width 0.5s;">
                                </div>
                                </div>
                                """, unsafe_allow_html=True)

                if is_churn:
                    st.warning("Attention : Risque de churn détecté.")
                else:
                    st.success("Client stable : Faible probabilité de churn.")
            else:
                st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")

