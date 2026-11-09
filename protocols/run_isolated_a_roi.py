"""Build leakage-isolated A/ROI meta-training probabilities, one outer fold.

Uses the original A stability, feature ranking, hyperparameter grid and
patient-level label alignment. Each invocation writes a fresh run directory.
"""

from __future__ import annotations

import argparse

import os

import hashlib

import json

import platform

import shutil

import sys

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

import sklearn

from sklearn.model_selection import StratifiedKFold

PROJECT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

PREVIOUS = Path(os.environ.get("CAP_COP_PROTOCOL_ROOT", Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(PREVIOUS))

import run_isolated_b_comparison as b

from cap_cop.pipelines import lesion_radiomics_replay_pipeline as a

MATRIX_PATH = PROJECT / "models/lesion_radiomics/locked_internal_perturbation_features.csv"

REF_PATH = PROJECT / "models/lesion_radiomics/internal_oof_predictions.csv"

XWALK_PATH = PROJECT / "data/public_subject_model_crosswalk.csv"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()

def load_inputs():
    clinical, _, _ = b.load_internal()
    xwalk = pd.read_csv(XWALK_PATH, dtype=str).query("cohort == 'internal'")
    if len(xwalk) != 375 or xwalk.public_subject_id.nunique() != 375 or xwalk.segmentation_model_id.nunique() != 375:
        raise ValueError("A segmentation crosswalk must cover 375 unique internal patients")
    labels = xwalk[["public_subject_id", "segmentation_model_id", "label_binary"]].rename(
        columns={"segmentation_model_id": "neutral_patient_id"}).merge(
        clinical[["public_subject_id", "outer_fold", "label_binary"]],
        on="public_subject_id", validate="one_to_one", suffixes=("_map", "_clinical"))
    if labels.label_binary_map.astype(int).ne(labels.label_binary_clinical.astype(int)).any():
        raise ValueError("A clinical/segmentation diagnosis mismatch")
    labels = labels.rename(columns={"label_binary_map": "label_binary"}).drop(columns=["label_binary_clinical"])
    matrix = pd.read_csv(MATRIX_PATH)
    matrix = matrix.merge(labels[["public_subject_id", "neutral_patient_id", "outer_fold"]],
                          on="neutral_patient_id", how="left", validate="many_to_one", sort=False,
                          suffixes=("_feature", "_lock"))
    if len(matrix) != 4875 or matrix.neutral_patient_id.nunique() != 375:
        raise ValueError("A perturbation matrix coverage differs from 375 x 13")
    if matrix.outer_fold_feature.astype(int).ne(matrix.outer_fold_lock.astype(int)).any():
        raise ValueError("A feature outer fold differs from locked shared fold")
    matrix = matrix.rename(columns={"outer_fold_lock": "outer_fold"}).drop(columns=["outer_fold_feature"])
    baseline = matrix.loc[matrix.perturbation.eq("BASELINE")].copy()
    X, y, folds = a.align_radiomics_training_inputs(labels, baseline, a.feature_cols(matrix))
    ref = pd.read_csv(REF_PATH)
    check = labels[["neutral_patient_id", "label_binary", "outer_fold"]].merge(
        ref, on="neutral_patient_id", validate="one_to_one", suffixes=("_label", "_ref"))
    if len(check) != 375 or check.label_binary_label.astype(int).ne(check.label_binary_ref.astype(int)).any() or check.outer_fold_label.astype(int).ne(check.outer_fold_ref.astype(int)).any():
        raise ValueError("A original OOF labels or folds differ from current patient contract")
    if not folds.astype(int).equals(labels.set_index("neutral_patient_id").loc[folds.index, "outer_fold"].astype(int)):
        raise ValueError("A aligned fold vector changed")
    return clinical, labels, matrix, X, y, folds, ref

def fit_a(matrix, X, y, train_ids, test_ids, context_fold, artifact_dir):
    train_ids = list(train_ids)
    test_ids = list(test_ids)
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids) or set(train_ids) & set(test_ids):
        raise ValueError("A fit/held-out IDs invalid")
    a.require_binary_classes(y.loc[train_ids], "A training")
    a.require_binary_classes(y.loc[test_ids], "A held-out")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    stability = a.stability_table(matrix, train_ids)
    stability.to_csv(artifact_dir / "outer_training_stability.csv", index=False)
    if not a.reduce_redundancy(X.loc[train_ids], stability):
        raise RuntimeError("A original algorithm found no robust features")
    scores = []
    Xtr = X.loc[train_ids]
    ytr = y.loc[train_ids]
    split = StratifiedKFold(4, shuffle=True, random_state=a.SEED + context_fold)
    for inner_fold, (itr, iva) in enumerate(split.split(Xtr, ytr)):
        inner_train_ids = Xtr.index[itr].tolist()
        inner_val_ids = Xtr.index[iva].tolist()
        a.require_binary_classes(y.loc[inner_train_ids], f"A hyperparameter inner {inner_fold} train")
        a.require_binary_classes(y.loc[inner_val_ids], f"A hyperparameter inner {inner_fold} validation")
        inner_stability = a.stability_table(matrix, inner_train_ids)
        features = a.reduce_redundancy(X.loc[inner_train_ids], inner_stability)
        if not features:
            continue
        ranked = a.rank_features(X.loc[inner_train_ids], y.loc[inner_train_ids], features)
        counts = sorted(set(n for n in [5, 10, 15, len(ranked)] if n <= len(ranked)))
        for n_features in counts:
            feature_set = ranked[:n_features]
            for kind, params in a.model_grid():
                model = a.build_model(kind, params)
                model.fit(X.loc[inner_train_ids, feature_set], y.loc[inner_train_ids])
                p = model.predict_proba(X.loc[inner_val_ids, feature_set])[:, 1]
                m = a.metrics(y.loc[inner_val_ids].to_numpy(), p)
                scores.append({"outer_fold": context_fold, "inner_fold": inner_fold,
                    "kind": kind, "params": json.dumps(params, sort_keys=True),
                    "n_features": n_features, "feature_set": "|".join(feature_set), **m})
        print(f"A context={context_fold} inner_hyper={inner_fold} complete", flush=True)
    if not scores:
        raise RuntimeError("A original hyperparameter grid produced no valid candidate")
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(artifact_dir / "hyperparameter_selection.csv", index=False)
    best = a.select_best_candidate(scores_df)
    selected_features = str(best.feature_set).split("|")
    selected_params = json.loads(str(best.params))
    model = a.build_model(str(best.kind), selected_params)
    model.fit(X.loc[train_ids, selected_features], y.loc[train_ids])
    predictions = model.predict_proba(X.loc[test_ids, selected_features])[:, 1]
    joblib.dump({"pipeline": model, "features": selected_features, "kind": best.kind,
                 "params": selected_params}, artifact_dir / "model.joblib")
    manifest = {"train_ids_sha256": ids_sha(train_ids), "test_ids_sha256": ids_sha(test_ids),
                "n_train": len(train_ids), "n_test": len(test_ids), "selected_features": selected_features,
                "selected_kind": str(best.kind), "selected_params": selected_params,
                "model_sha256": sha(artifact_dir / "model.joblib"),
                "candidate_table_sha256": sha(artifact_dir / "hyperparameter_selection.csv")}
    (artifact_dir / "fit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return predictions, manifest

def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing A run: {output}")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "run_source_snapshot.py")
    clinical, labels, matrix, X, y, folds, ref = load_inputs()
    source_sha = {str(p): sha(p) for p in [Path(__file__), Path(a.__file__), Path(b.clinical.__file__),
        MATRIX_PATH, REF_PATH, XWALK_PATH]}
    (output / "input_manifest.json").write_text(json.dumps({"source_sha256": source_sha,
        "external_labels_read": False, "environment": {"python": platform.python_version(),
        "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outer = args.outer
    outer_train_ids = folds.index[folds.ne(outer)].tolist()
    outer_held_ids = folds.index[folds.eq(outer)].tolist()
    if len(outer_train_ids) != 300 or len(outer_held_ids) != 75:
        raise ValueError("A shared outer fold expected 300/75 patients")
    patients = labels.set_index("neutral_patient_id").public_subject_id
    if args.reference_only:
        p, _ = fit_a(matrix, X, y, outer_train_ids, outer_held_ids, outer, output / "outer_reference_refit")
        expected = ref.set_index("neutral_patient_id").loc[outer_held_ids, "probability_COP"].to_numpy()
        delta = float(np.max(np.abs(p - expected)))
        pd.DataFrame({"neutral_patient_id": outer_held_ids, "public_subject_id": patients.loc[outer_held_ids].to_numpy(),
                      "label": y.loc[outer_held_ids].to_numpy(dtype=int), "probability_COP": p,
                      "original_probability_COP": expected}).to_csv(output / "reference_refit_predictions.csv", index=False)
        (output / "reference_refit_check.json").write_text(json.dumps({"max_abs_delta": delta,
            "refit_auroc": float(a.metrics(y.loc[outer_held_ids].to_numpy(dtype=int), p)["AUROC"]),
            "reference_auroc": float(a.metrics(y.loc[outer_held_ids].to_numpy(dtype=int), expected)["AUROC"]),
            "status": "MATCH" if delta <= 1e-7 else "DIFFERENT"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"A reference-only fold={outer} max delta={delta}", flush=True)
        return
    clinical_train = clinical.loc[clinical.outer_fold.ne(outer)]
    clinical_held = clinical.loc[clinical.outer_fold.eq(outer)]
    y_clinical = clinical_train.label_binary.astype(int).to_numpy()
    p_clinical_sorted = b.clinical_inner_oof(clinical_train, y_clinical, outer)
    p_clinical_inner = pd.Series(p_clinical_sorted, index=clinical_train.public_subject_id).loc[
        patients.loc[outer_train_ids]].to_numpy()
    p_image_inner = np.full(300, np.nan)
    outer_y = y.loc[outer_train_ids].to_numpy(dtype=int)
    inner_split = StratifiedKFold(5, shuffle=True, random_state=20260909 + outer)
    for inner, (itr, iva) in enumerate(inner_split.split(np.zeros(300), outer_y)):
        if args.inner is not None and inner != args.inner:
            continue
        fit_ids = [outer_train_ids[i] for i in itr]
        val_ids = [outer_train_ids[i] for i in iva]
        if set(fit_ids) & set(outer_held_ids):
            raise ValueError("A meta-held-out patient entered inner image fit")
        p, manifest = fit_a(matrix, X, y, fit_ids, val_ids, inner, output / f"meta_inner_{inner}")
        p_image_inner[iva] = p
        pd.DataFrame({"neutral_patient_id": val_ids, "public_subject_id": patients.loc[val_ids].to_numpy(),
                      "label": y.loc[val_ids].to_numpy(dtype=int), "p_image": p,
                      "meta_outer_fold": outer, "meta_inner_fold": inner}).to_csv(
            output / f"meta_inner_{inner}" / "heldout_predictions.csv", index=False)
        print(f"A meta_outer={outer} meta_inner={inner} complete", flush=True)
    if args.inner is not None:
        print("A single-inner pilot complete; no fusion weight selected", flush=True)
        return
    if not np.isfinite(p_image_inner).all():
        raise ValueError("A inner image OOF incomplete")
    meta_train = pd.DataFrame({"neutral_patient_id": outer_train_ids,
        "public_subject_id": patients.loc[outer_train_ids].to_numpy(), "label": outer_y,
        "p_clinical": p_clinical_inner, "p_image": p_image_inner})
    meta_train.to_csv(output / "meta_training_oof.csv", index=False)
    weight, _, grid = b.select_weight_from_df(meta_train[["label", "p_clinical", "p_image"]])
    grid.to_csv(output / "weight_grid.csv", index=False)
    p_clinical_held_sorted = b.clinical_heldout(clinical_train, clinical_held, y_clinical, outer)
    p_clinical_held = pd.Series(p_clinical_held_sorted, index=clinical_held.public_subject_id).loc[
        patients.loc[outer_held_ids]].to_numpy()
    p_image_held, manifest = fit_a(matrix, X, y, outer_train_ids, outer_held_ids, outer, output / "outer_heldout")
    reference = ref.set_index("neutral_patient_id").loc[outer_held_ids, "probability_COP"].to_numpy()
    delta = float(np.max(np.abs(p_image_held - reference)))
    held = pd.DataFrame({"neutral_patient_id": outer_held_ids,
        "public_subject_id": patients.loc[outer_held_ids].to_numpy(), "label": y.loc[outer_held_ids].to_numpy(dtype=int),
        "outer_fold": outer, "p_clinical": p_clinical_held, "p_image": p_image_held,
        "p_image_historical_reference": reference,
        "weight_clinical": weight, "p_fusion": weight * p_clinical_held + (1 - weight) * p_image_held})
    held.to_csv(output / "outer_heldout_predictions.csv", index=False)
    (output / "outer_result.json").write_text(json.dumps({"weight_clinical": float(weight),
        "historical_image_max_delta": delta, "historical_image_match": bool(delta <= 1e-7),
        "outer_metrics": b.metrics(held.label.to_numpy(), held.p_fusion.to_numpy())},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"A outer={outer} complete, weight={weight:.2f}, historical max delta={delta:.3g}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--outer", type=int, choices=range(5), required=True)
    parser.add_argument("--inner", type=int, choices=range(5))
    parser.add_argument("--reference-only", action="store_true")
    run(parser.parse_args())
