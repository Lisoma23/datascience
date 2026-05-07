Phase 1 -- Setup & EDA

- [x] Initialiser le projet (venv, git, structure de dossiers)
- [x] Charger et auditer le dataset
- [x] EDA complète (distributions, corrélations, déséquilibre classes, outliers)

Phase 2 -- Preprocessing & Feature Engineering

- [x] Pipeline sklearn (imputation, encoding, scaling)
- [x] Feature engineering (ratios, interactions)
- [x] Split train/test stratifié

Phase 3 -- Modélisation

- [x] Baseline (Logistic Regression)
- [x] Random Forest
- [x] Gradient Boosting (sklearn)
- [x] Cross-validation stratifiée + gestion déséquilibre (SMOTE, class_weight, seuil)

Phase 4 -- Évaluation & Interprétabilité

- [x] Comparaison structurée (tableau, graphes)
- [x] Analyse d'erreurs (matrice de confusion, profiling FN)
- [x] Feature Importance + SHAP
- [x] Sélection et justification du modèle final

Phase 5 -- API REST

- [x] FastAPI avec /predict et /health
- [x] /model-info (métadonnées du modèle)
- [x] Sérialisation du modèle (joblib)
- [x] Validation des inputs, gestion d'erreurs

Phase 6 -- Dashboard

- [ ] Streamlit orienté métier (KPIs, prédiction temps réel, simulation scénarios)
- [ ] Appelle l'API (pas le modèle directement)

Phase 7 -- Rapport & Présentation

- [ ] Rapport 6 pages max
- [ ] Support de présentation
