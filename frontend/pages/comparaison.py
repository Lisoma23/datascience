import streamlit as st
import pandas as pd
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

metrics_path = ROOT_DIR / "artifacts" / "metrics.json"

st.title("Comparaison des Modèles")

if metrics_path.exists():
    with open(metrics_path, "r") as f:
        data = json.load(f)
    
    # filtre pour ne garder que les modèles
    models_data = {k: v for k, v in data.items() if not k.startswith('_')}
    
    # création du DataFrame
    df_metrics = pd.DataFrame.from_dict(models_data, orient='index').reset_index()
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

confusion_matrix_path = ROOT_DIR / "reports" / "figures" / "model_comparison.png"


st.markdown("### Matrices de Confusion")

if confusion_matrix_path.exists():
    st.image(
        str(confusion_matrix_path), 
        caption="Analyse comparative : Logistic Regression (gauche), Random Forest (milieu) et Gradient Boosting (droite)",
        width='stretch'
    )
else:
    # Placeholder si l'image n'existe pas 
    st.image("https://placehold.co/1000x400?text=ROC+and+PR+Curves+Placeholder", width='stretch')