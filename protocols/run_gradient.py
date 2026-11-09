"""Controlled A-image / clinical-input comparison; local research runner."""

from __future__ import annotations

import os

import argparse

import hashlib

import json

import platform

import sys

from datetime import datetime, timezone

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

import sklearn

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score

from sklearn.model_selection import StratifiedKFold

PROJECT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

COMPARISON = Path(os.environ.get("CAP_COP_PROTOCOL_ROOT", Path(__file__).resolve().parent))

PROJECT_REPORT = Path(os.environ.get("CAP_COP_PROTOCOL_ROOT", Path(__file__).resolve().parent))

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(COMPARISON)]

from cap_cop.clinical import clinical_missingness_simulation as v30

from cap_cop.clinical import clinical_transportability_protocol as v3

from cap_cop.clinical import full_structured_input_strategy_experiment as clinical

from cap_cop.data.release_inputs import load_clinical_cohort

import run_isolated_b_comparison as comparison

A_DIR = PROJECT / "outputs/strict_meta_a_roi_result_sklearn161_20261029_01"

A_INTERNAL = A_DIR / "A_strict_internal_oof.csv"

A_EXTERNAL = A_DIR / "A_strict_external_predictions.csv"

A_FOLDS = [PROJECT / "outputs" / ("strict_meta_a_sklearn161_20261029_outer_1_v2" if k == 1 else f"strict_meta_a_sklearn161_20261029_outer_{k}") for k in range(5)]

XWALK = PROJECT / "data/public_subject_model_crosswalk.csv"

GRADIENTS = {
    "core7": list(v3.CORE7),
    "core9_hscrp_pct": list(v3.CORE7_HSCRP_PCT),
    "transportable11": list(v3.TRANSPORTABLE11),
    "full16": list(v3.ALL_FEATURES),
}

SEEDS = list(clinical.SEEDS)

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def id_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()

def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

def check_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = list(GRADIENTS.values())
    if any(not set(levels[i]).issubset(levels[i + 1]) for i in range(3)):
        raise ValueError("Historical clinical feature sets are not nested")
    if levels[-1] != list(clinical.FEATURE_SETS["full16"]):
        raise ValueError("Full input differs from original A clinical contract")
    internal = pd.read_csv(A_INTERNAL, dtype={"public_subject_id": str})
    truth, _, _ = comparison.load_internal()
    truth = truth[["public_subject_id", "label_binary", "outer_fold", *GRADIENTS["full16"]]]
    merged = truth.merge(internal[["public_subject_id", "label", "outer_fold", "p_clinical", "p_image", "p_fusion"]],
                         on="public_subject_id", validate="one_to_one", suffixes=("", "_a"))
    if len(merged) != 375 or merged.public_subject_id.nunique() != 375 or int(merged.label_binary.sum()) != 125:
        raise ValueError("A internal patient coverage invalid")
    if merged.label_binary.ne(merged.label).any() or merged.outer_fold.ne(merged.outer_fold_a).any():
        raise ValueError("A internal label or fold differs from clinical source")
    if set(merged.outer_fold.astype(int)) != set(range(5)):
        raise ValueError("A outer fold IDs invalid")
    baseline = comparison.metrics(merged.label.to_numpy(dtype=int), merged.p_fusion.to_numpy(dtype=float))["AUROC"]
    if abs(baseline - 0.854304) > 1e-5:
        raise ValueError("A strict full16 baseline no longer matches its audited result")
    return truth, internal

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int, path: Path) -> np.ndarray:
    if set(train.public_subject_id) & set(test.public_subject_id):
        raise ValueError("Clinical train and held-out IDs overlap")
    y = train.label_binary.to_numpy(dtype=int)
    if set(y) != {0, 1} or set(test.label_binary.astype(int)) != {0, 1}:
        raise ValueError("Clinical fit or evaluation has one class")
    train_clean = v30.clean(train[features], features)
    test_clean = v30.clean(test[features], features)
    imputer = v30.FoldImputer(features, "median", seed).fit(train_clean)
    train_design = v30.make_design(imputer.transform(train_clean), train_clean, features)
    test_design = v30.make_design(imputer.transform(test_clean), test_clean, features)
    augmented, augmented_y = clinical.augment_training_design(train_design, y, features, "balanced_bootstrap", seed + 7919)
    fitted = v30.fit_one_model(augmented, augmented_y, features, "random_forest", seed)
    pred = v30.predict_one_model(fitted, test_design, features)
    if len(pred) != len(test) or not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
        raise ValueError("Clinical probabilities invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": imputer, "fitted": fitted, "features": features, "seed": seed}, path)
    write_json(path.with_suffix(".json"), {"train_n": len(train), "test_n": len(test),
               "train_id_sha256": id_sha(train.public_subject_id), "test_id_sha256": id_sha(test.public_subject_id),
               "train_labels": {"CAP": int((y == 0).sum()), "COP": int((y == 1).sum())},
               "model_sha256": sha(path), "seed": seed})
    return pred

def clinical_inner_oof(frame: pd.DataFrame, features: list[str], outer: int, output: Path) -> np.ndarray:
    y = frame.label_binary.to_numpy(dtype=int)
    members = []
    for seed in SEEDS:
        pred = np.full(len(frame), np.nan)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for k, (train_idx, val_idx) in enumerate(cv.split(frame, y)):
            pred[val_idx] = fit_predict(frame.iloc[train_idx], frame.iloc[val_idx], features,
                                        seed * 100 + k, output / f"seed_{seed}_inner_{k}.joblib")
        if not np.isfinite(pred).all():
            raise ValueError("Clinical meta training OOF incomplete")
        members.append(pred)
    return np.mean(np.vstack(members), axis=0)

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    q = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return {"AUROC": float(roc_auc_score(y, p)), "Accuracy": float(accuracy_score(y, q)),
            "Sensitivity": float(tp / (tp + fn)), "Specificity": float(tn / (tn + fp)),
            "Precision": float(precision_score(y, q, zero_division=0)), "F1": float(f1_score(y, q, zero_division=0)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def internal_phase(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite gradient run: {out}")
    truth, baseline = check_source()
    out.mkdir(parents=True)
    write_json(out / "run_contract.json", {
        "status": "NEW_POST_HOC_NESTED_SUBSETS_FROM_HISTORICAL_FEATURE_DEFINITIONS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "external_labels_read": False,
        "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "seeds": SEEDS, "clinical_model": "full_structured_input_strategy_experiment RF balanced_bootstrap",
        "fusion": "locked_weighted_multimodal_fusion_pipeline.select_weight_from_df",
        "feature_sets": GRADIENTS,
        "input_sha256": {str(path): sha(path) for path in [A_INTERNAL, XWALK,
            PROJECT / "src/cap_cop/clinical/full_structured_input_strategy_experiment.py",
            PROJECT / "src/cap_cop/clinical/clinical_transportability_protocol.py",
            PROJECT / "src/cap_cop/clinical/clinical_missingness_simulation.py",
            PROJECT / "src/cap_cop/pipelines/locked_weighted_multimodal_fusion_pipeline.py", Path(__file__)]},
    })
    predictions = []
    selection = []
    for level, features in GRADIENTS.items():
        if level == "full16":
            frame = baseline[["public_subject_id", "label", "outer_fold", "p_clinical", "p_image", "p_fusion", "weight_clinical"]].copy()
            frame["level"] = level
            predictions.append(frame)
            selection.append({"level": level, "fold_weights": [float(baseline.loc[baseline.outer_fold.eq(k), "weight_clinical"].iloc[0]) for k in range(5)]})
            continue
        rows = []
        for outer in range(5):
            train = truth.loc[truth.outer_fold.ne(outer)].copy()
            held = truth.loc[truth.outer_fold.eq(outer)].copy()
            if len(train) != 300 or len(held) != 75:
                raise ValueError("A outer split differs from 300/75")
            image_source = pd.read_csv(A_FOLDS[outer] / "meta_training_oof.csv", dtype={"public_subject_id": str})
            image_train = train[["public_subject_id", "label_binary"]].merge(
                image_source[["public_subject_id", "label", "p_image"]], on="public_subject_id", validate="one_to_one")
            if len(image_train) != 300 or image_train.label_binary.ne(image_train.label).any():
                raise ValueError("A training-side image predictions or labels invalid")
            held_image = held[["public_subject_id", "label_binary"]].merge(
                baseline[["public_subject_id", "label", "p_image"]], on="public_subject_id", validate="one_to_one")
            if held_image.label_binary.ne(held_image.label).any() or len(held_image) != 75:
                raise ValueError("A held-out image identity mismatch")
            fold_dir = out / level / f"outer_{outer}"
            p_meta_clinical = clinical_inner_oof(train, features, outer, fold_dir / "meta_clinical_models")
            meta = train[["public_subject_id", "label_binary"]].copy()
            meta["p_clinical"] = p_meta_clinical
            meta = meta.merge(image_train[["public_subject_id", "p_image"]], on="public_subject_id", validate="one_to_one")
            meta = meta.rename(columns={"label_binary": "label"})
            weight, tie, grid = comparison.select_weight_from_df(meta[["label", "p_clinical", "p_image"]])
            fold_dir.mkdir(parents=True, exist_ok=True)
            meta.to_csv(fold_dir / "meta_training_oof.csv", index=False)
            grid.to_csv(fold_dir / "weight_grid.csv", index=False)
            write_json(fold_dir / "weight_selection.json", {"weight_clinical": weight, "tie": tie,
                       "outer_train_id_sha256": id_sha(train.public_subject_id), "outer_held_id_sha256": id_sha(held.public_subject_id),
                       "source_image_meta_sha256": sha(A_FOLDS[outer] / "meta_training_oof.csv")})
            held_members = [fit_predict(train, held, features, seed * 100 + outer,
                            fold_dir / "heldout_clinical_models" / f"seed_{seed}.joblib") for seed in SEEDS]
            pred = held[["public_subject_id", "label_binary", "outer_fold"]].copy()
            pred["p_clinical"] = np.mean(np.vstack(held_members), axis=0)
            pred = pred.merge(held_image[["public_subject_id", "p_image"]], on="public_subject_id", validate="one_to_one")
            pred = pred.rename(columns={"label_binary": "label"})
            pred["p_fusion"] = weight * pred.p_clinical + (1 - weight) * pred.p_image
            pred["weight_clinical"] = weight
            pred.to_csv(fold_dir / "heldout_predictions.csv", index=False)
            rows.append(pred)
            print(f"{level} outer={outer} weight={weight:.2f}", flush=True)
        frame = pd.concat(rows, ignore_index=True)
        frame["level"] = level
        predictions.append(frame)
    all_preds = pd.concat(predictions, ignore_index=True).sort_values(["level", "public_subject_id"])
    if len(all_preds) != 4 * 375 or all_preds.groupby("level").public_subject_id.nunique().ne(375).any():
        raise ValueError("Gradient internal OOF coverage incomplete")
    all_preds.to_csv(out / "internal_predictions.csv", index=False)
    locks = {}
    for level, frame in all_preds.groupby("level"):
        if level == "full16":
            reference = json.loads((A_DIR / "internal_weight_lock.json").read_text())
            weight = float(reference["external_clinical_weight"])
            grid = pd.read_csv(A_DIR / "A_external_weight_grid_internal_only.csv")
        else:
            weight, _, grid = comparison.select_weight_from_df(frame[["label", "p_clinical", "p_image"]])
        grid.to_csv(out / f"{level}_external_weight_grid.csv", index=False)
        locks[level] = {"external_clinical_weight": weight,
                        "fold_weights": {str(k): float(v) for k, v in frame.groupby("outer_fold").weight_clinical.first().items()},
                        "internal": {branch: metrics(frame.label.to_numpy(int), frame[column].to_numpy(float)) for branch, column in
                                     [("clinical", "p_clinical"), ("image", "p_image"), ("fusion", "p_fusion")]}}
    write_json(out / "internal_weight_lock.json", {"external_labels_read": False,
               "internal_predictions_sha256": sha(out / "internal_predictions.csv"), "levels": locks})
    print("Internal gradient and weight lock complete", flush=True)

def external_phase(out: Path) -> None:
    lock = json.loads((out / "internal_weight_lock.json").read_text())
    if lock["external_labels_read"] is not False or sha(out / "internal_predictions.csv") != lock["internal_predictions_sha256"]:
        raise ValueError("Internal-only weight lock changed")
    if (out / "external_predictions.csv").exists():
        raise FileExistsError("External evaluation already completed")
    truth, _ = check_source()
    external = load_clinical_cohort("external")
    image = pd.read_csv(A_EXTERNAL, dtype={"public_subject_id": str})
    merged = external.merge(image[["public_subject_id", "label", "p_image", "p_clinical", "p_fusion"]],
                            on="public_subject_id", validate="one_to_one", suffixes=("", "_a"))
    if len(merged) != 33 or merged.public_subject_id.nunique() != 33 or int(merged.label_binary.sum()) != 14:
        raise ValueError("External gradient cohort invalid")
    if merged.label_binary.ne(merged.label).any() or set(merged.public_subject_id) & set(truth.public_subject_id):
        raise ValueError("External patient labels or internal overlap invalid")
    rows = []
    for level, features in GRADIENTS.items():
        if level == "full16":
            frame = merged[["public_subject_id", "label", "p_clinical", "p_image", "p_fusion"]].copy()
            if abs(float(lock["levels"][level]["external_clinical_weight"]) - 0.75) > 1e-12:
                raise ValueError("Full16 A external weight lock differs")
        else:
            members = []
            for index, seed in enumerate(SEEDS):
                masked = clinical.scenario_frame(truth[features], merged[features], features, f"external_mask_{index+1}", seed)
                train_frame = truth[["public_subject_id", "label_binary", *features]].copy()
                train_frame.loc[:, features] = masked.to_numpy()
                members.append(fit_predict(train_frame, merged[["public_subject_id", "label_binary", *features]],
                                           features, seed, out / level / "external_clinical_models" / f"seed_{seed}.joblib"))
            frame = merged[["public_subject_id", "label", "p_image"]].copy()
            frame["p_clinical"] = np.mean(np.vstack(members), axis=0)
            w = float(lock["levels"][level]["external_clinical_weight"])
            frame["p_fusion"] = w * frame.p_clinical + (1 - w) * frame.p_image
        frame["level"] = level
        rows.append(frame)
        print(f"{level} external evaluated", flush=True)
    result = pd.concat(rows, ignore_index=True).sort_values(["level", "public_subject_id"])
    result.to_csv(out / "external_predictions.csv", index=False)
    write_json(out / "external_result.json", {"external_predictions_sha256": sha(out / "external_predictions.csv"),
        "external_image_sha256": sha(A_EXTERNAL), "internal_weight_lock_sha256": sha(out / "internal_weight_lock.json"),
        "levels": {level: {branch: metrics(frame.label.to_numpy(int), frame[column].to_numpy(float)) for branch, column in
                   [("clinical", "p_clinical"), ("image", "p_image"), ("fusion", "p_fusion")]} for level, frame in result.groupby("level")}})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["internal", "external"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (internal_phase if args.phase == "internal" else external_phase)(args.output.resolve())
