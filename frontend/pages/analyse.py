from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

feature_importance_path = ROOT_DIR / "reports" / "figures" / "feature_importance.png"
shap_summary_path = ROOT_DIR / "reports" / "figures" / "shap_summary.png"


st.title(" Analyse de l'Importance des Variables")
st.subheader("Quels facteurs influencent le plus le départ de vos clients ?")

st.markdown("### Importance Globale (Feature Importance)")
st.write("""
    Ce graphique montre l'importance relative de chaque variable dans les décisions du modèle.
    Plus la barre est longue, plus la variable a un impact sur la prédiction finale.
""")


if feature_importance_path.exists():
    st.image(feature_importance_path, caption="Classement des variables par importance")
else:
    # Placeholder si l'image n'existe pas
    st.image("https://placehold.co/800x400?text=Feature+Importance+Placeholder", width="stretch")

st.divider()

st.markdown("### Analyse SHAP (Interprétabilité Locale & Globale)")
st.write("""
    Le graphique SHAP explique comment chaque valeur impacte positivement ou négativement la probabilité de churn.
    *Le rouge indique une valeur élevée de la variable, le bleu une valeur basse.*
""")


if shap_summary_path.exists():
    st.image(shap_summary_path, caption="SHAP Summary Plot")
else:
    # Placeholder si l'image n'existe pas
    st.image("https://placehold.co/800x400?text=SHAP+Summary+Placeholder", width="stretch")

st.divider()

st.markdown("### Explications des variables clés")

col1, col2 = st.columns(2)

with col1:
    st.write("**Facteurs de Risque (Churn ↑)**")
    st.markdown("""
    * **Contrats mensuels** : Les clients sans engagement à long terme sont plus volatiles.
    * **Échecs de paiement** : Un incident financier est souvent le premier signe d'un départ.
    * **Tickets de support élevés** : Une frustration technique non résolue augmente radicalement le risque.
    """)

with col2:
    st.write("**Facteurs de Rétention (Churn ↓)**")
    st.markdown("""
    * **Ancienneté (Tenure)** : Plus un client reste longtemps, plus il est fidèle.
    * **Nombre de fonctionnalités utilisées** : Un client qui adopte l'outil en profondeur est plus "collant".
    * **Score CSAT élevé** : La satisfaction déclarée reste un rempart solide contre le départ.
    """)
