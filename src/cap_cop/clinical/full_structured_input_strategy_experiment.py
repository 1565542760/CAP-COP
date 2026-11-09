# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import numpy as np

import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC

from cap_cop.clinical import clinical_missingness_simulation as v30

from cap_cop.clinical import clinical_transportability_protocol as v3

SEEDS = [20260908, 20260909, 20260910]

N_SPLITS = 5

FEATURE_SETS = {
    "full16": v3.ALL_FEATURES,
    "transportable15_no_CRP": [
        feature for feature in v3.ALL_FEATURES if feature != "c_reactive_protein"
    ],
    "transportable11": v3.TRANSPORTABLE11,
    "core7_hscrp_pct": v3.CORE7_HSCRP_PCT,
    "core7": v3.CORE7,
}

JITTER_SCALE = 0.05

def scenario_frame(
    frame: pd.DataFrame,
    external: pd.DataFrame,
    features: list[str],
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    if scenario == "normal":
        return frame.copy()
    return v30.apply_external_mask(frame, external, features, scenario, seed)

def augment_training_design(
    design: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    mode: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Augment only an already-imputed training fold."""
    if mode == "none":
        return design.copy(), np.asarray(y, dtype=int).copy()

    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    class_indices = {
        cls: np.flatnonzero(y == cls)
        for cls in np.unique(y)
    }
    target_n = max(len(indices) for indices in class_indices.values())

    categorical = [
        feature for feature in features if feature in v30.CATEGORICAL
    ]
    continuous = [
        column for column in design.columns
        if column not in categorical
        and not column.startswith("__missing_")
    ]

    rows = [design.copy()]
    labels = [y.copy()]
    for cls, indices in sorted(class_indices.items()):
        if len(indices) >= target_n:
            continue
        sampled = rng.choice(indices, size=target_n - len(indices), replace=True)
        synthetic = design.iloc[sampled].copy().reset_index(drop=True)

        if mode == "balanced_jitter" and continuous:
            for column in continuous:
                values = pd.to_numeric(design.iloc[indices][column], errors="coerce")
                scale = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                if not np.isfinite(scale) or scale <= 0:
                    continue
                noise = rng.normal(
                    loc=0.0,
                    scale=JITTER_SCALE * scale,
                    size=len(synthetic),
                )
                lower = float(values.min())
                upper = float(values.max())
                synthetic[column] = (
                    pd.to_numeric(synthetic[column], errors="coerce")
                    + noise
                ).clip(lower=lower, upper=upper)

        rows.append(synthetic)
        labels.append(np.full(len(synthetic), cls, dtype=int))

    augmented = pd.concat(rows, ignore_index=True)
    augmented_y = np.concatenate(labels)
    order = rng.permutation(len(augmented))
    return augmented.iloc[order].reset_index(drop=True), augmented_y[order]

def fit_predict_one(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    model_spec: str,
    augmentation: str,
    seed: int,
) -> np.ndarray:
    train_clean = v30.clean(train_frame, features)
    eval_clean = v30.clean(eval_frame, features)
    imputer = v30.FoldImputer(features, "median", seed)
    imputer.fit(train_clean)
    train_imputed = imputer.transform(train_clean)
    eval_imputed = imputer.transform(eval_clean)
    train_design = v30.make_design(train_imputed, train_clean, features)
    eval_design = v30.make_design(eval_imputed, eval_clean, features)
    augmented_design, augmented_y = augment_training_design(
        train_design,
        y_train,
        features,
        augmentation,
        seed + 7919,
    )
    if model_spec == "svm":
        categorical = [
            feature for feature in features if feature in v30.CATEGORICAL
        ]
        numeric = [
            column
            for column in augmented_design.columns
            if column not in categorical
        ]
        encoded_train = pd.get_dummies(
            augmented_design,
            columns=categorical,
            drop_first=False,
            dtype=float,
        )
        encoded_eval = pd.get_dummies(
            eval_design,
            columns=categorical,
            drop_first=False,
            dtype=float,
        ).reindex(columns=encoded_train.columns, fill_value=0.0)
        scaler = RobustScaler()
        matrix_train = scaler.fit_transform(encoded_train)
        matrix_eval = scaler.transform(encoded_eval)
        model = SVC(
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        )
        model.fit(matrix_train, augmented_y)
        return model.predict_proba(matrix_eval)[:, 1]

    fitted = v30.fit_one_model(
        augmented_design,
        augmented_y,
        features,
        model_spec,
        seed,
    )
    return v30.predict_one_model(fitted, eval_design, features)

def cross_fitted(
    X: pd.DataFrame,
    external: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    model_spec: str,
    augmentation: str,
    scenario: str,
    seed: int,
) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    masked = scenario_frame(X, external, features, scenario, seed)
    predictions = np.full(len(masked), np.nan, dtype=float)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    for fold, (train_idx, val_idx) in enumerate(splitter.split(masked, y)):
        predictions[val_idx] = fit_predict_one(
            masked.iloc[train_idx],
            masked.iloc[val_idx],
            y[train_idx],
            features,
            model_spec,
            augmentation,
            seed * 100 + fold,
        )
    return predictions

def external_predictions(
    X: pd.DataFrame,
    external: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    model_spec: str,
    augmentation: str,
) -> np.ndarray:
    predictions = []
    for index, seed in enumerate(SEEDS):
        train_frame = scenario_frame(
            X,
            external,
            features,
            f"external_mask_{index + 1}",
            seed,
        )
        predictions.append(
            fit_predict_one(
                train_frame,
                external,
                y,
                features,
                model_spec,
                augmentation,
                seed,
            )
        )
    return np.mean(np.vstack(predictions), axis=0)
