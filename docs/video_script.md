# Script Vidéo — Démonstration du Projet

**Durée cible : 12-15 min**
**Format : enregistrement écran + voix, pas de slides**

---

## PARTIE 1 — Introduction & Contexte (Louisan, ~1 min 30)

**Ecran : README du repo GitHub**

> "Bonjour, nous sommes Louisan, Sofia, Miriam et Faustine. On vous présente notre projet : un système intelligent de rétention client basé sur le Machine Learning."

> "Le problème business : dans le SaaS, perdre un client coûte 5 à 7 fois plus cher que d'en retenir un. On a un dataset de 10 000 clients avec 32 variables — démographie, usage, paiement, satisfaction — et une variable cible : le churn, à environ 10%."

> "Notre objectif : prédire quels clients vont partir, et fournir un outil décisionnel complet — modèle, API, dashboard."

---

## PARTIE 2 — Preprocessing & Feature Engineering (Miriam, ~2 min 30)

**Ecran 1 : `notebooks/02_preprocessing.ipynb` (ou le notebook EDA)**

> "Avant de modéliser, on a préparé les données. Le dataset a pas de valeurs manquantes sauf `complaint_type` qui est nullable — c'est normal, un client sans réclamation n'a pas de type."

**Ecran 2 : `backend/data/preprocessing.py` — montrer le code**

> "On a créé un module Python réutilisable, pas juste un notebook. Ici on voit les features engineered qu'on a ajoutées :"

- `revenue_per_month` = total_revenue / tenure_months
- `support_rate` = support_tickets / tenure_months
- `engagement_score` = monthly_logins * weekly_active_days

> "On a séparé les colonnes numériques, catégorielles et ordinales. Le preprocessing est un `ColumnTransformer` sklearn avec StandardScaler pour le numérique, OneHotEncoder pour le catégoriel, et OrdinalEncoder pour les variables ordonnées."

> "Le split est stratifié à 80/20 pour préserver le ratio de churn. Pas de data leakage : le scaler est fit uniquement sur le train."

**Ecran 3 : Terminal — lancer `pytest tests/test_preprocessing.py -v`**

> "On a 26 tests unitaires pour le preprocessing. Tout passe."

**Ecran 4 : `.github/workflows/ci.yml`**

> "On a mis en place un CI avec GitHub Actions : ruff pour le linting et pytest pour les tests, qui tournent à chaque push."

---

## PARTIE 3 — Modélisation & Résultats (Louisan, ~3 min 30)

**Ecran 1 : `notebooks/03_modeling.ipynb` — cellules de comparaison**

> "On a entraîné et comparé 3 modèles : Logistic Regression, Random Forest, et Gradient Boosting. Le sujet demande minimum 3."

> "Le dataset est déséquilibré — 90% non-churn, 10% churn. Un modèle naïf qui prédit toujours 'non-churn' a 90% d'accuracy mais 0% de recall. L'accuracy est donc trompeuse ici."

**Ecran 2 : Tableau comparatif des métriques (dans le notebook)**

> "Voici les résultats au seuil par défaut de 0.50 :"

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | ~82% | ~0.35 | ~0.65 | ~0.45 | ~0.79 |
| Random Forest | ~89% | 0 | 0 | 0 | ~0.73 |
| Gradient Boosting | ~89% | ~0.50 | ~0.05 | ~0.10 | ~0.80 |

> "Random Forest a 89% d'accuracy mais ne détecte aucun churner — precision et recall à 0. C'est exactement le piège dont on parlait."

> "Le ROC-AUC, qui est indépendant du seuil, montre que le Gradient Boosting est le meilleur à 0.80."

**Ecran 3 : Courbe de seuil / F1 dans le notebook**

> "On a appliqué SMOTE pour rééquilibrer le jeu d'entraînement, puis on a optimisé le seuil de décision. Au lieu de 0.50, notre seuil optimal est 0.10 — ce qui maximise le F1."

> "Pourquoi un seuil aussi bas ? Parce que le dataset est synthétique avec des corrélations faibles — maximum |r| = 0.16. Les probabilités du modèle sont naturellement basses, entre 1% et 25%. A 0.50, on ne détecte rien."

**Ecran 4 : SHAP summary plot (dans le notebook ou `reports/figures/`)**

> "On a utilisé SHAP pour l'interprétabilité. Les variables les plus influentes sont : les échecs de paiement, l'ancienneté du client, le type de contrat, et le score de satisfaction."

> "Par exemple, un client avec beaucoup d'échecs de paiement et un contrat mensuel a un risque de churn beaucoup plus élevé."

---

## PARTIE 4 — API FastAPI (Sofia, ~2 min 30)

**Ecran 1 : Terminal — lancer l'API**

```bash
make api
```

> "On a développé une API REST avec FastAPI. Elle expose le modèle comme un service, exactement comme en production."

**Ecran 2 : Navigateur — `http://localhost:8000/docs` (Swagger)**

> "La documentation est auto-générée. On a 3 endpoints :"

**Ecran 3 : Tester `/health`**

> "GET /health vérifie que le service est actif et que le modèle est chargé. On voit `model_loaded: true`."

**Ecran 4 : Tester `/predict` avec un payload**

> "POST /predict reçoit les données d'un client en JSON et renvoie la prédiction."

Utiliser ce payload de test :
```json
{
  "gender": "Male",
  "age": 45,
  "country": "UK",
  "city": "London",
  "customer_segment": "Individual",
  "tenure_months": 3,
  "signup_channel": "Web",
  "contract_type": "Monthly",
  "monthly_logins": 2,
  "weekly_active_days": 1,
  "avg_session_time": 5.0,
  "features_used": 1,
  "usage_growth_rate": -0.3,
  "last_login_days_ago": 20,
  "monthly_fee": 80.0,
  "total_revenue": 240.0,
  "payment_method": "Card",
  "payment_failures": 3,
  "discount_applied": "No",
  "price_increase_last_3m": "Yes",
  "support_tickets": 5,
  "avg_resolution_time": 48.0,
  "complaint_type": "Billing",
  "csat_score": 1.5,
  "escalations": 2,
  "email_open_rate": 0.05,
  "marketing_click_rate": 0.01,
  "nps_score": -50,
  "survey_response": "Unsatisfied",
  "referral_count": 0
}
```

> "Ce client a un profil à risque : contrat mensuel, 3 échecs de paiement, satisfaction très basse, usage en chute. Le modèle retourne une probabilité de churn élevée et `churn_prediction: 1`."

**Ecran 5 : Tester `/model-info`**

> "GET /model-info renvoie les métriques de comparaison des 3 modèles, le nom du modèle actif et son seuil. C'est cet endpoint que le dashboard utilise pour afficher la page Comparaison."

---

## PARTIE 5 — Dashboard Streamlit (Faustine, ~3 min 30)

**Ecran 1 : Terminal — lancer le dashboard**

```bash
make dashboard
```

> "Le dashboard est notre interface décisionnelle. Il s'adresse à un responsable marketing ou CRM."

**Ecran 2 : Page "Dashboard" — KPIs + graphiques**

> "La page principale montre les KPIs : nombre total de clients, taux de churn, revenu mensuel total, et revenu à risque."

> "En dessous, on a 4 graphiques : la répartition churn vs non-churn, le churn par segment client, le churn par type de contrat, et la satisfaction par statut."

> "En bas, la section 'Clients à haut risque'. On peut choisir combien de clients analyser avec le slider. Le dashboard appelle l'API `/predict` pour chaque client et affiche les 10 plus risqués avec leur score, segment, ancienneté et satisfaction."

**Ecran 3 : Page "Prédiction"**

> "Ici un utilisateur métier peut saisir manuellement le profil d'un client — âge, contrat, usage, satisfaction — et obtenir la probabilité de churn en temps réel. Le formulaire appelle directement l'API."

Remplir le formulaire avec un profil à risque et montrer le résultat.

**Ecran 4 : Page "Comparaison des modèles"**

> "Cette page affiche le tableau comparatif des 3 modèles avec leurs métriques. Les données viennent de l'endpoint `/model-info` de l'API. On voit les courbes ROC et Precision-Recall si les images sont disponibles."

**Ecran 5 : Page "Simulateur de Scénarios"**

> "Le simulateur permet de tester l'impact d'un changement. Par exemple : que se passe-t-il si on accorde une remise à un client ? Si on améliore son score de satisfaction ?"

Montrer : modifier le CSAT de 2 à 4 dans la sidebar et montrer la variation du risque.

> "On voit le risque initial vs le risque simulé, et l'impact du scénario — positif, négatif ou nul."

**Ecran 6 : Page "Analyse"**

> "Enfin, la page d'analyse montre l'importance des variables et le SHAP summary plot, avec des explications métier : quels facteurs augmentent le risque, quels facteurs protègent le client."

---

## PARTIE 6 — Conclusion (Louisan, ~30 sec)

**Ecran : Dashboard ou repo GitHub**

> "Pour résumer : on a un pipeline complet — du preprocessing au dashboard — avec 3 modèles comparés, un seuil optimisé, des tests automatisés en CI, une API REST, et un dashboard interactif. Le tout est reproductible : `make install`, `make train`, `make api`, `make dashboard`."

> "Merci pour votre attention."

---

## Checklist avant enregistrement

- [ ] `make api` fonctionne (API tourne sur localhost:8000)
- [ ] `make dashboard` fonctionne (Streamlit tourne sur localhost:8501)
- [ ] Les artifacts sont présents dans `artifacts/` (model.joblib, threshold.json, metrics.json)
- [ ] Les figures sont dans `reports/figures/` (feature_importance.png, shap_summary.png, roc_pr_curves.png)
- [ ] Tester le payload de la partie 4 avant l'enregistrement
- [ ] Chaque membre a lu et répété sa partie
- [ ] Enregistrement écran + audio (OBS / QuickTime / Zoom)
