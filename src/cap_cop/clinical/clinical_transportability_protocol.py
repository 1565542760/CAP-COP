# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

ALL_FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "weight_loss",
    "lymphocyte_count",
    "erythrocyte_sedimentation_rate",
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
]

CORE7 = [
    "age",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "weight_loss",
    "lymphocyte_count",
]

CORE7_HSCRP_PCT = CORE7 + [
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
]

TRANSPORTABLE11 = [
    "age",
    "height_cm",
    "weight_kg",
    "body_mass_index",
    "fever",
    "cough",
    "sputum_category",
    "weight_loss",
    "lymphocyte_count",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
]

LAB_FEATURES = {
    "c_reactive_protein",
    "high_sensitivity_c_reactive_protein",
    "procalcitonin",
    "erythrocyte_sedimentation_rate",
}

class FrameCleaner(BaseEstimator, TransformerMixin):
    """Apply only label-blind, deterministic data cleaning inside each fold."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in ["height_cm", "weight_kg", "body_mass_index"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X.loc[X[col] == 0, col] = np.nan
        if {"height_cm", "weight_kg", "body_mass_index"} <= set(X.columns):
            valid_bmi = (
                X["body_mass_index"].isna()
                & X["height_cm"].notna()
                & X["weight_kg"].notna()
                & (X["height_cm"] > 0)
            )
            X.loc[valid_bmi, "body_mass_index"] = X.loc[valid_bmi, "weight_kg"] / (
                (X.loc[valid_bmi, "height_cm"] / 100.0) ** 2
            )
        return X
