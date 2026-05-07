"""
Module de preprocessing réutilisable.

Usage depuis la racine du projet (datascience/) :
    from backend.data.preprocessing import load_and_split, build_preprocessor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TARGET = "churn"
ID_COL = "customer_id"

# 19 colonnes numériques brutes (hors target et id)
NUM_COLS: list[str] = [
    "age",
    "tenure_months",
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "usage_growth_rate",
    "last_login_days_ago",
    "monthly_fee",
    "total_revenue",
    "payment_failures",
    "support_tickets",
    "avg_resolution_time",
    "csat_score",
    "escalations",
    "email_open_rate",
    "marketing_click_rate",
    "nps_score",
    "referral_count",
]

# Catégorielles nominales (OneHot) — inclut les binaires discount_applied
# et price_increase_last_3m qui n'ont pas d'ordre naturel.
CAT_COLS_ONEHOT: list[str] = [
    "gender",
    "country",
    "city",
    "signup_channel",
    "payment_method",
    "complaint_type",
    "survey_response",
    "discount_applied",
    "price_increase_last_3m",
]

# Catégorielles ordinales (ordre explicite)
CAT_COLS_ORDINAL: list[str] = [
    "customer_segment",
    "contract_type",
]

CAT_COLS: list[str] = CAT_COLS_ONEHOT + CAT_COLS_ORDINAL

ENGINEERED_COLS: list[str] = [
    "revenue_per_month",
    "tickets_per_tenure",
    "login_intensity",
]

DEFAULT_CSV = Path(__file__).resolve().parent.parent.parent / "customer_churn.csv"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features dérivées. Retourne une copie enrichie."""
    df = df.copy()

    df["revenue_per_month"] = np.where(
        df["tenure_months"] > 0,
        df["total_revenue"] / df["tenure_months"],
        df["monthly_fee"],
    )

    df["tickets_per_tenure"] = np.where(
        df["tenure_months"] > 0,
        df["support_tickets"] / df["tenure_months"],
        0,
    )

    df["login_intensity"] = np.where(
        df["weekly_active_days"] > 0,
        df["monthly_logins"] / df["weekly_active_days"],
        0,
    )

    return df


# ---------------------------------------------------------------------------
# Preprocessor (ColumnTransformer)
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Retourne un ColumnTransformer prêt à fit sur le train set."""

    all_num_cols = NUM_COLS + ENGINEERED_COLS

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    onehot_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="No_Complaint")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_num_cols),
            ("onehot", onehot_pipeline, CAT_COLS_ONEHOT),
            ("ordinal", ordinal_pipeline, CAT_COLS_ORDINAL),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Load & split
# ---------------------------------------------------------------------------


def load_and_split(
    csv_path: str | Path = DEFAULT_CSV,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Charge le CSV, applique le feature engineering, split stratifié."""

    df = pd.read_csv(csv_path)
    df = add_engineered_features(df)

    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train : {X_train.shape} | Churn rate: {y_train.mean():.1%}")
    print(f"Test  : {X_test.shape}  | Churn rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Feature names utilitaire
# ---------------------------------------------------------------------------


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retourne les noms de features après transformation."""

    all_num_cols = NUM_COLS + ENGINEERED_COLS

    onehot_names = list(
        preprocessor.named_transformers_["onehot"]["encoder"].get_feature_names_out(
            CAT_COLS_ONEHOT
        )
    )

    return all_num_cols + onehot_names + CAT_COLS_ORDINAL
