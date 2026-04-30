"""Tests unitaires pour backend.data.preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from backend.data.preprocessing import (
    CAT_COLS_ONEHOT,
    CAT_COLS_ORDINAL,
    ENGINEERED_COLS,
    ID_COL,
    NUM_COLS,
    TARGET,
    add_engineered_features,
    build_preprocessor,
    get_feature_names,
    load_and_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_df():
    """Charge le dataset brut."""
    return pd.read_csv("customer_churn.csv")


@pytest.fixture()
def split_data():
    """Retourne X_train, X_test, y_train, y_test."""
    return load_and_split("customer_churn.csv")


@pytest.fixture()
def fitted_preprocessor(split_data):
    """Retourne (preprocessor, X_train_processed, X_test_processed)."""
    X_train, X_test, _, _ = split_data
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    return preprocessor, X_train_t, X_test_t


# ---------------------------------------------------------------------------
# add_engineered_features
# ---------------------------------------------------------------------------


class TestAddEngineeredFeatures:
    def test_returns_dataframe(self, raw_df):
        result = add_engineered_features(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_expected_columns(self, raw_df):
        result = add_engineered_features(raw_df)
        for col in ENGINEERED_COLS:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_does_not_mutate_input(self, raw_df):
        original_cols = list(raw_df.columns)
        add_engineered_features(raw_df)
        assert list(raw_df.columns) == original_cols

    def test_no_nan_in_engineered(self, raw_df):
        result = add_engineered_features(raw_df)
        for col in ENGINEERED_COLS:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_division_by_zero_handled(self):
        """Quand tenure_months=0 ou weekly_active_days=0, pas de NaN/Inf."""
        df = pd.DataFrame(
            {
                "tenure_months": [0, 5],
                "total_revenue": [100, 500],
                "monthly_fee": [50, 100],
                "support_tickets": [2, 3],
                "monthly_logins": [10, 20],
                "weekly_active_days": [0, 4],
            }
        )
        result = add_engineered_features(df)
        assert np.isfinite(result["revenue_per_month"]).all()
        assert np.isfinite(result["tickets_per_tenure"]).all()
        assert np.isfinite(result["login_intensity"]).all()

    def test_revenue_per_month_correctness(self):
        df = pd.DataFrame(
            {
                "tenure_months": [10],
                "total_revenue": [500],
                "monthly_fee": [50],
                "support_tickets": [0],
                "monthly_logins": [10],
                "weekly_active_days": [5],
            }
        )
        result = add_engineered_features(df)
        assert result["revenue_per_month"].iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# load_and_split
# ---------------------------------------------------------------------------


class TestLoadAndSplit:
    def test_returns_four_elements(self, split_data):
        assert len(split_data) == 4

    def test_shapes(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        assert X_train.shape[0] == 8000
        assert X_test.shape[0] == 2000
        assert y_train.shape[0] == 8000
        assert y_test.shape[0] == 2000

    def test_stratification(self, split_data):
        """Le taux de churn doit etre quasi identique dans train et test."""
        _, _, y_train, y_test = split_data
        assert abs(y_train.mean() - y_test.mean()) < 0.01

    def test_no_target_in_X(self, split_data):
        X_train, X_test, _, _ = split_data
        assert TARGET not in X_train.columns
        assert TARGET not in X_test.columns

    def test_no_id_in_X(self, split_data):
        X_train, X_test, _, _ = split_data
        assert ID_COL not in X_train.columns
        assert ID_COL not in X_test.columns

    def test_engineered_cols_present(self, split_data):
        X_train, _, _, _ = split_data
        for col in ENGINEERED_COLS:
            assert col in X_train.columns

    def test_deterministic(self):
        """Deux appels identiques donnent le meme split."""
        X1, _, y1, _ = load_and_split("customer_churn.csv")
        X2, _, y2, _ = load_and_split("customer_churn.csv")
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_expected_transformers(self):
        preprocessor = build_preprocessor()
        names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in names
        assert "onehot" in names
        assert "ordinal" in names

    def test_fit_transform_shape(self, split_data):
        X_train, _, _, _ = split_data
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(X_train)
        assert result.shape[0] == X_train.shape[0]
        assert result.shape[1] > X_train.shape[1]  # OneHot expands columns

    def test_no_nan_after_transform(self, fitted_preprocessor):
        _, X_train_t, X_test_t = fitted_preprocessor
        assert not np.isnan(X_train_t).any(), "NaN in transformed train"
        assert not np.isnan(X_test_t).any(), "NaN in transformed test"

    def test_scaling_mean_std(self, fitted_preprocessor):
        """Les features numeriques doivent etre centrees-reduites."""
        preprocessor, X_train_t, _ = fitted_preprocessor
        feature_names = get_feature_names(preprocessor)
        n_num = len(NUM_COLS + ENGINEERED_COLS)
        num_data = X_train_t[:, :n_num]
        means = num_data.mean(axis=0)
        stds = num_data.std(axis=0)
        np.testing.assert_allclose(means, 0, atol=0.01)
        np.testing.assert_allclose(stds, 1, atol=0.05)

    def test_no_data_leakage(self, split_data):
        """Le preprocessor fit sur train ne doit pas voir le test."""
        X_train, X_test, _, _ = split_data
        p1 = build_preprocessor()
        p1.fit(X_train)

        # Transformer le test avec le preprocessor fit sur train
        X_test_t = p1.transform(X_test)

        # Les moyennes du test transforme ne doivent PAS etre exactement 0
        # (sinon ca voudrait dire qu'on a fit sur le test)
        n_num = len(NUM_COLS + ENGINEERED_COLS)
        test_means = X_test_t[:, :n_num].mean(axis=0)
        assert not np.allclose(test_means, 0, atol=0.001), \
            "Test means are ~0 — possible data leakage"


# ---------------------------------------------------------------------------
# get_feature_names
# ---------------------------------------------------------------------------


class TestGetFeatureNames:
    def test_returns_list(self, fitted_preprocessor):
        preprocessor, _, _ = fitted_preprocessor
        names = get_feature_names(preprocessor)
        assert isinstance(names, list)

    def test_length_matches_columns(self, fitted_preprocessor):
        preprocessor, X_train_t, _ = fitted_preprocessor
        names = get_feature_names(preprocessor)
        assert len(names) == X_train_t.shape[1]

    def test_contains_num_cols(self, fitted_preprocessor):
        preprocessor, _, _ = fitted_preprocessor
        names = get_feature_names(preprocessor)
        for col in NUM_COLS + ENGINEERED_COLS:
            assert col in names, f"Missing numeric feature: {col}"

    def test_contains_ordinal_cols(self, fitted_preprocessor):
        preprocessor, _, _ = fitted_preprocessor
        names = get_feature_names(preprocessor)
        for col in CAT_COLS_ORDINAL:
            assert col in names, f"Missing ordinal feature: {col}"


# ---------------------------------------------------------------------------
# Constants coherence
# ---------------------------------------------------------------------------


class TestConstants:
    def test_no_overlap_num_cat(self):
        all_cat = set(CAT_COLS_ONEHOT + CAT_COLS_ORDINAL)
        all_num = set(NUM_COLS)
        overlap = all_cat & all_num
        assert not overlap, f"Overlap between num and cat: {overlap}"

    def test_target_not_in_features(self):
        all_cols = NUM_COLS + CAT_COLS_ONEHOT + CAT_COLS_ORDINAL + ENGINEERED_COLS
        assert TARGET not in all_cols

    def test_id_not_in_features(self):
        all_cols = NUM_COLS + CAT_COLS_ONEHOT + CAT_COLS_ORDINAL + ENGINEERED_COLS
        assert ID_COL not in all_cols
