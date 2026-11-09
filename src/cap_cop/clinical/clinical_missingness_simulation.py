# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn.linear_model import BayesianRidge, LogisticRegression

from sklearn.preprocessing import RobustScaler

from cap_cop.clinical import clinical_transportability_protocol as v3

CATEGORICAL = {
    "fever",
    "cough",
    "sputum_category",
    "chest_tightness",
    "dyspnoea",
    "haemoptysis",
    "weight_loss",
}

def clean(frame, features):
    return v3.FrameCleaner().fit_transform(frame[features].copy())

def apply_external_mask(frame, external, features, scenario, seed):
    out = frame.copy()
    if scenario == "normal":
        return out
    pattern = external[features].isna().to_numpy()
    rng = np.random.default_rng(seed + int(scenario.rsplit("_", 1)[1]))
    sampled = pattern[rng.integers(0, len(pattern), size=len(out))]
    for index, feature in enumerate(features):
        out.loc[sampled[:, index], feature] = np.nan
    return out

class FoldImputer:
    def __init__(self, features, method, seed):
        self.features = list(features)
        self.method = method
        self.seed = seed

    def fit(self, frame):
        self.categories = {}
        clean_frame = clean(frame, self.features)
        values = clean_frame[self.features].apply(pd.to_numeric, errors="coerce")
        self.medians = {
            feature: float(values[feature].median())
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        self.lower = {
            feature: float(values[feature].quantile(0.01))
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        self.upper = {
            feature: float(values[feature].quantile(0.99))
            if values[feature].notna().any()
            else 0.0
            for feature in self.features
        }
        for feature in self.features:
            if feature in CATEGORICAL:
                observed = sorted(values[feature].dropna().unique().tolist())
                self.categories[feature] = observed or [0.0]
        if self.method == "mice_bayesian_posterior":
            self.imputer = IterativeImputer(
                estimator=BayesianRidge(),
                initial_strategy="median",
                max_iter=20,
                tol=1e-3,
                sample_posterior=True,
                keep_empty_features=True,
                random_state=self.seed,
            )
            self.imputer.fit(values.to_numpy(dtype=float, copy=True))
        else:
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            self.imputer.fit(values.to_numpy(dtype=float, copy=True))
        return self

    def transform(self, frame, imputation_index=0):
        clean_frame = clean(frame, self.features)
        values = clean_frame[self.features].apply(pd.to_numeric, errors="coerce")
        if self.method == "mice_bayesian_posterior":
            if imputation_index:
                self.imputer.random_state = self.seed + imputation_index
            imputed = self.imputer.transform(
                values.to_numpy(dtype=float, copy=True)
            )
        else:
            imputed = self.imputer.transform(
                values.to_numpy(dtype=float, copy=True)
            )
        result = pd.DataFrame(imputed, columns=self.features, index=frame.index)
        for feature in self.features:
            result[feature] = result[feature].clip(
                lower=self.lower[feature],
                upper=self.upper[feature],
            )
            if feature in self.categories:
                categories = np.asarray(self.categories[feature])
                result[feature] = result[feature].map(
                    lambda value: float(categories[np.argmin(np.abs(categories - value))])
                )
        return result

def make_design(imputed, original, features):
    result = imputed.copy()
    missing = original[features].isna()
    for feature in features:
        result[f"__missing_{feature}"] = missing[feature].astype(float)
    clinical = [
        feature
        for feature in ["fever", "cough", "sputum_category", "weight_loss"]
        if feature in features
    ]
    labs = [feature for feature in v3.LAB_FEATURES if feature in features]
    anthropometric = [
        feature
        for feature in ["height_cm", "weight_kg", "body_mass_index"]
        if feature in features
    ]
    result["__missing_count_selected"] = missing.sum(axis=1).astype(float)
    result["__missing_count_clinical"] = missing[clinical].sum(axis=1).astype(float)
    result["__missing_count_laboratory"] = missing[labs].sum(axis=1).astype(float)
    result["__missing_count_anthropometric"] = missing[anthropometric].sum(axis=1).astype(float)
    return result

def fit_one_model(train_design, y, features, model_spec, seed):
    categorical = [feature for feature in features if feature in CATEGORICAL]
    numeric = [column for column in train_design.columns if column not in categorical]
    encoded_train = pd.get_dummies(
        train_design,
        columns=categorical,
        drop_first=False,
        dtype=float,
    )
    if model_spec == "logistic":
        scaler = RobustScaler()
        matrix = scaler.fit_transform(encoded_train[numeric + [
            column for column in encoded_train.columns if column not in numeric
        ]])
        model = LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
        )
        model.fit(matrix, y)
        return ("logistic", model, scaler, encoded_train.columns.tolist())
    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(encoded_train, y)
    return ("random_forest", model, None, encoded_train.columns.tolist())

def predict_one_model(fitted, design, features):
    kind, model, scaler, columns = fitted
    categorical = [feature for feature in features if feature in CATEGORICAL]
    encoded = pd.get_dummies(design, columns=categorical, drop_first=False, dtype=float)
    encoded = encoded.reindex(columns=columns, fill_value=0.0)
    if kind == "logistic":
        return model.predict_proba(scaler.transform(encoded))[:, 1]
    return model.predict_proba(encoded)[:, 1]
