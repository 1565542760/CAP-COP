# Final-method dependency subset; retained scientific definitions are source-verbatim.
"""Validate patient-keyed labels and immutable fold metadata before model fitting."""

from pathlib import Path

import pandas as pd

def require_unique_ids(values: pd.Series, context: str) -> None:
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"{context}: empty patient identifiers")
    if values.duplicated().any():
        raise ValueError(f"{context}: duplicate patient identifiers")

def require_binary_classes(labels: pd.Series, context: str) -> None:
    counts = labels.value_counts(dropna=False).to_dict()
    if labels.isna().any() or set(counts) != {0, 1}:
        raise ValueError(f"{context}: expected both binary classes; observed counts={counts}")

def align_radiomics_training_inputs(labels: pd.DataFrame, baseline: pd.DataFrame,
                                    features: list[str]):
    key = "neutral_patient_id"
    require_unique_ids(labels[key], "Clinical labels")
    require_unique_ids(baseline[key], "Baseline radiomics")
    if set(labels[key]) != set(baseline[key]):
        raise ValueError("Baseline radiomics and clinical labels have different patient ID sets")
    y = pd.to_numeric(labels.set_index(key)["label_binary"], errors="raise")
    require_binary_classes(y, "Internal cohort")
    x = baseline.set_index(key)[features].apply(pd.to_numeric, errors="coerce")
    # Explicit ID lookup retains feature-row order; no positional relabelling.
    y = y.loc[x.index].astype(int)
    folds = pd.to_numeric(baseline.set_index(key)["outer_fold"], errors="raise")
    if folds.isna().any() or set(folds) != set(range(5)):
        raise ValueError("Expected the five locked outer folds numbered 0 through 4")
    folds = folds.astype(int)
    if "label_binary" in baseline:
        supplied = pd.to_numeric(baseline.set_index(key)["label_binary"], errors="raise")
        if not supplied.eq(y).all():
            raise ValueError("Baseline labels conflict with the patient-keyed clinical labels")
    for fold in range(5):
        require_binary_classes(y.loc[folds.ne(fold)], f"Outer {fold} training")
        require_binary_classes(y.loc[folds.eq(fold)], f"Outer {fold} validation")
    return x, y, folds

def read_stable_cohort_metadata(path: Path, *, legacy: bool = False) -> pd.DataFrame:
    """Read only identities, labels, folds and domains, never prediction columns."""
    key = "neutral_imaging_id"
    fold = "fold_or_domain" if legacy else "outer_fold"
    columns = [key, "label", fold, "primary_domain"]
    if legacy:
        frame = pd.read_csv(path, usecols=[*columns, "candidate"], dtype={key: str})
        frame = frame.loc[frame["candidate"].eq("R0_GENESIS_B2_BASELINE"), columns]
    else:
        frame = pd.read_csv(path, usecols=columns, dtype={key: str})
    # Exact repeated metadata rows can occur in a multi-candidate source table.
    # Conflicting rows for a patient must never be resolved by taking the first.
    frame = frame.drop_duplicates().rename(columns={fold: "outer_fold"})
    require_unique_ids(frame[key], "Stable representation cohort")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    require_binary_classes(frame["label"], "Stable representation cohort")
    frame["outer_fold"] = pd.to_numeric(frame["outer_fold"], errors="raise")
    if frame["outer_fold"].isna().any() or not frame["outer_fold"].mod(1).eq(0).all():
        raise ValueError("Cohort metadata contains invalid fold numbers")
    if frame["primary_domain"].isna().any() or frame["primary_domain"].astype(str).str.strip().eq("").any():
        raise ValueError("Cohort metadata contains empty domains")
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    frame["label"] = frame["label"].astype(int)
    if frame["outer_fold"].nunique() != 5:
        raise ValueError("Expected five locked outer folds")
    for fold_id in frame["outer_fold"].unique():
        require_binary_classes(frame.loc[frame["outer_fold"].ne(fold_id), "label"], "Stable outer training")
        require_binary_classes(frame.loc[frame["outer_fold"].eq(fold_id), "label"], "Stable outer validation")
    return frame

def stable_patient_contract(base: pd.DataFrame, external_index: pd.DataFrame,
                            crosswalk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bind distinct imaging namespaces to one public identity before feature joins."""
    required = {"public_subject_id", "cohort", "label_binary", "segmentation_model_id",
                "lesion_imaging_model_id", "whole_chest_imaging_model_id", "external_tensor_id"}
    if required.difference(crosswalk):
        raise ValueError(f"Identity contract lacks columns: {sorted(required.difference(crosswalk))}")
    parts = {}
    for cohort in ("internal", "external"):
        part = crosswalk.loc[crosswalk["cohort"].eq(cohort)].copy()
        if part.empty:
            raise ValueError(f"Identity contract: no {cohort} patients")
        for key in ("public_subject_id", "segmentation_model_id", "lesion_imaging_model_id",
                    "whole_chest_imaging_model_id"):
            require_unique_ids(part[key], f"{cohort} {key}")
        part["label_binary"] = pd.to_numeric(part["label_binary"], errors="raise").astype(int)
        parts[cohort] = part
    if set(parts["internal"]["public_subject_id"]) & set(parts["external"]["public_subject_id"]):
        raise ValueError("Identity contract: internal and external patients overlap")
    require_unique_ids(base["neutral_imaging_id"], "Stable internal whole-chest IDs")
    require_unique_ids(external_index["neutral_external_id"], "Stable external tensor IDs")
    require_unique_ids(external_index["neutral_patient_id"], "Stable external segmentation IDs")
    require_unique_ids(parts["external"]["external_tensor_id"], "External tensor crosswalk")

    internal = base.merge(parts["internal"], left_on="neutral_imaging_id",
                          right_on="whole_chest_imaging_model_id", how="left",
                          validate="one_to_one", sort=False, indicator=True)
    if len(internal) != len(parts["internal"]) or not internal["_merge"].eq("both").all():
        raise ValueError("Stable internal whole-chest IDs do not exactly match the crosswalk")
    if not pd.to_numeric(internal["label"], errors="raise").eq(internal["label_binary"]).all():
        raise ValueError("Stable internal labels disagree with the patient crosswalk")

    external = external_index.merge(parts["external"], left_on=["neutral_external_id", "neutral_patient_id"],
                                    right_on=["external_tensor_id", "segmentation_model_id"],
                                    how="left", validate="one_to_one", sort=False, indicator=True)
    if len(external) != len(parts["external"]) or not external["_merge"].eq("both").all():
        raise ValueError("Stable external tensor/segmentation ID pairs do not exactly match the crosswalk")
    return internal.drop(columns="_merge"), external.drop(columns="_merge")

def align_stable_patient_features(target: pd.DataFrame, source: pd.DataFrame,
                                  features: list[str], *, cohort: str,
                                  check_lesion_id: bool = False,
                                  check_external_tensor: bool = False) -> pd.DataFrame:
    """Align source features by segmentation ID and verify all available identity evidence."""
    if not features or len(features) != len(set(features)) or set(features).difference(source):
        raise ValueError(f"{cohort} source: missing or duplicate feature names")
    require_unique_ids(source["neutral_patient_id"], f"{cohort} source segmentation IDs")
    joined = target.merge(source, left_on="segmentation_model_id", right_on="neutral_patient_id",
                          how="left", validate="one_to_one", sort=False, indicator=True,
                          suffixes=("", "_source"))
    if len(joined) != len(source) or not joined["_merge"].eq("both").all():
        raise ValueError(f"{cohort} source: missing or orphan segmentation identities")
    checks = (("public_subject_id_source", "public_subject_id"),
              ("label_binary_source", "label_binary"),
              ("outer_fold_source", "outer_fold"))
    for supplied, expected in checks:
        if supplied in joined and expected in joined:
            if expected in {"label_binary", "outer_fold"}:
                actual_values = pd.to_numeric(joined[supplied], errors="raise")
                expected_values = pd.to_numeric(joined[expected], errors="raise")
            else:
                actual_values = joined[supplied].astype(str)
                expected_values = joined[expected].astype(str)
            if not actual_values.eq(expected_values).all():
                raise ValueError(f"{cohort} source: {expected} disagrees with patient identity")
    if check_lesion_id:
        if "source_neutral_imaging_id" not in joined or not joined["source_neutral_imaging_id"].astype(str).eq(joined["lesion_imaging_model_id"].astype(str)).all():
            raise ValueError(f"{cohort} source: lesion imaging ID disagrees with crosswalk")
    if check_external_tensor:
        if "neutral_external_id_source" in joined:
            actual = joined["neutral_external_id_source"]
        else:
            actual = joined.get("neutral_external_id")
        if actual is None or not actual.astype(str).eq(joined["external_tensor_id"].astype(str)).all():
            raise ValueError(f"{cohort} source: external tensor ID disagrees with crosswalk")
    return joined
