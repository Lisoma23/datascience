import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
metrics_path = ROOT_DIR / "artifacts" / "metrics.json"
url = "http://127.0.0.1:8000/model-info"

st.title("Comparaison des Modèles")

# --- FONCTION DE RÉCUPÉRATION DES DONNÉES ---
def fetch_model_metrics():
    """Tentative de récupération de données..."""

    try:
        response = requests.get(url)
        if response.status_code == 200:
            st.caption("Données chargées")
            return response.json()
    except Exception:
        pass

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            st.caption("API indisponible : Chargement depuis l'artifact local")
            return json.load(f)

    return None

# --- AFFICHAGE DU TABLEAU ---
data = fetch_model_metrics()

if data and 'metrics' in data:
    actual_metrics = data['metrics']

    models_only = {k: v for k, v in actual_metrics.items() if not k.startswith('_')}

    df_metrics = pd.DataFrame.from_dict(models_only, orient='index').reset_index()
    df_metrics.rename(columns={'index': 'Modèle'}, inplace=True)

    # highlight des valeurs importantes
    st.dataframe(
        df_metrics.style.highlight_max(axis=0, subset=['accuracy', 'f1', 'roc_auc'], color='#28A745'),
        width='stretch'
    )
else:
    st.error("Impossible de trouver le fichier dans artifacts")
    st.info(f"Chemin vérifié : {metrics_path}")

    dummy_metrics = {
        "Modèle": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Précision": [0.82, 0.88, 0.91]
    }
    st.dataframe(pd.DataFrame(dummy_metrics), width='stretch')

st.divider()

roc_pr_curves_path = ROOT_DIR / "reports" / "figures" / "roc_pr_curves.png"

st.markdown("### Courbes ROC & Precision-Recall")

if roc_pr_curves_path.exists():
    st.image(
        str(roc_pr_curves_path),
        caption="Analyse comparative : Courbe ROC (gauche) et Precision-Recall (droite)",
        width='stretch'
    )
else:
    # Placeholder si l'image n'existe pas
    st.image("https://placehold.co/1000x400?text=ROC+and+PR+Curves+Placeholder", width='stretch')

st.divider()

# Comparaison des modèles
st.markdown("### Comparaison Globale")
bar_chart_comparison_path = ROOT_DIR / "backend" / "reports" / "figures" / "model_comparison.png"

if bar_chart_comparison_path.exists():
    st.image(
        str(bar_chart_comparison_path),
        caption="Performance relative des modèles",
        width="stretch"
    )
else:
    st.info("Graphique de comparaison indisponible")
