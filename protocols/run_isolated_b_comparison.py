"""Restricted, leakage-isolated fusion audit for the reported B comparator.

Internal and external phases are separate so no external label is opened before
the internal weight policy has been written and hashed.
"""

from __future__ import annotations

import argparse

import os

import hashlib

import json

import sys

from datetime import datetime, timezone

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

SRC = Path(__file__).resolve().parents[1] / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cap_cop.clinical import full_structured_input_strategy_experiment as clinical

from cap_cop.data.release_inputs import load_clinical_cohort

from cap_cop.pipelines.locked_weighted_multimodal_fusion_pipeline import select_weight_from_df

OUTPUT = ROOT / "outputs" / "formal_comparison_meta_isolation_20261028_b_v3"

C_OOF = ROOT / "outputs/model_c_correct_identity_formal_20261028_1518/model_c_internal_oof_verified.csv"

XWALK = ROOT / "data/public_subject_model_crosswalk.csv"

B_DIR = ROOT / "models/whole_chest_ct"

REQUIRED_CLASSES = {0, 1}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def require_binary(y: np.ndarray, where: str) -> None:
    if set(map(int, y)) != REQUIRED_CLASSES:
        raise ValueError(f"{where}: both CAP=0 and COP=1 are required")

def make_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
            class_weight="balanced", random_state=20260909,
        ),
    )

def fit_probe(X: np.ndarray, y: np.ndarray, rows: np.ndarray, embedding_index: np.ndarray):
    # The released B replay fits in embedding-manifest order. The solver can
    # change slightly when the same training rows are presented in another order.
    ordered = rows[np.argsort(embedding_index[rows], kind="mergesort")]
    return make_probe().fit(X[ordered], y[ordered])

def load_internal() -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    clinical_frame = load_clinical_cohort("internal")
    folds = pd.read_csv(C_OOF)[["public_subject_id", "label", "outer_fold"]]
    xwalk = pd.read_csv(XWALK, dtype=str)
    xwalk = xwalk.loc[xwalk.cohort.eq("internal"), ["public_subject_id", "whole_chest_imaging_model_id"]]
    manifest = pd.read_csv(B_DIR / "internal_embedding_manifest.csv")
    locked_b = pd.read_csv(B_DIR / "internal_oof_predictions.csv")
    merged = clinical_frame.merge(folds, on="public_subject_id", validate="one_to_one")
    merged = merged.merge(xwalk, on="public_subject_id", validate="one_to_one")
    merged = merged.merge(
        manifest[["neutral_imaging_id", "embedding_index"]],
        left_on="whole_chest_imaging_model_id", right_on="neutral_imaging_id", validate="one_to_one",
    )
    merged = merged.merge(
        locked_b[["neutral_imaging_id", "label", "fold_id"]], on="neutral_imaging_id",
        suffixes=("", "_b"), validate="one_to_one",
    )
    if len(merged) != 375 or merged.public_subject_id.nunique() != 375:
        raise ValueError("Expected 375 one-to-one internal patient rows")
    if not np.array_equal(merged.label_binary.astype(int), merged.label.astype(int)):
        raise ValueError("Clinical/C diagnosis mismatch")
    if not np.array_equal(merged.label_binary.astype(int), merged.label_b.astype(int)):
        raise ValueError("Clinical/B diagnosis mismatch")
    if not np.array_equal(merged.outer_fold.astype(int), merged.fold_id.astype(int)):
        raise ValueError("B/C outer folds differ; unified fold cannot be assumed")
    if sorted(merged.outer_fold.unique().tolist()) != list(range(5)):
        raise ValueError("Expected outer folds 0..4")
    embeddings = np.load(B_DIR / "embeddings/internal_embeddings.npy", mmap_mode="r")
    idx = merged.embedding_index.astype(int).to_numpy()
    if len(set(idx)) != 375 or idx.min() != 0 or idx.max() != 374:
        raise ValueError("Embedding rows are not a permutation of 0..374")
    if embeddings.shape[0] != 375 or not np.isfinite(embeddings[idx]).all():
        raise ValueError("B embedding matrix invalid")
    merged = merged.sort_values("public_subject_id", kind="mergesort").reset_index(drop=True)
    matrix = np.asarray(embeddings[merged.embedding_index.astype(int).to_numpy()], dtype=np.float32)
    required = {
        str(C_OOF): sha(C_OOF), str(XWALK): sha(XWALK),
        str(B_DIR / "internal_embedding_manifest.csv"): sha(B_DIR / "internal_embedding_manifest.csv"),
        str(B_DIR / "internal_oof_predictions.csv"): sha(B_DIR / "internal_oof_predictions.csv"),
        str(B_DIR / "embeddings/internal_embeddings.npy"): sha(B_DIR / "embeddings/internal_embeddings.npy"),
    }
    return merged, matrix, required

def clinical_inner_oof(frame: pd.DataFrame, y: np.ndarray, outer: int) -> np.ndarray:
    features = clinical.FEATURE_SETS["full16"]
    X = frame[features].reset_index(drop=True)
    require_binary(y, f"outer {outer} meta-train")
    members = [
        clinical.cross_fitted(X, X, y, features, "random_forest", "balanced_bootstrap", "normal", seed)
        for seed in clinical.SEEDS
    ]
    result = np.mean(np.vstack(members), axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Clinical inner OOF incomplete")
    return result

def clinical_heldout(train: pd.DataFrame, held: pd.DataFrame, y: np.ndarray, outer: int) -> np.ndarray:
    features = clinical.FEATURE_SETS["full16"]
    members = [
        clinical.fit_predict_one(
            train[features], held[features], y, features,
            "random_forest", "balanced_bootstrap", seed * 100 + outer,
        )
        for seed in clinical.SEEDS
    ]
    return np.mean(np.vstack(members), axis=0)

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    q = (p >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "Accuracy": float(accuracy_score(y, q)),
        "Sensitivity": float(q[y == 1].mean()),
        "Specificity": float((q[y == 0] == 0).mean()),
        "F1": float(f1_score(y, q)),
        "TN": int(((y == 0) & (q == 0)).sum()),
        "FP": int(((y == 0) & (q == 1)).sum()),
        "FN": int(((y == 1) & (q == 0)).sum()),
        "TP": int(((y == 1) & (q == 1)).sum()),
    }

def internal_phase() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite prior run: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    frame, X, hashes = load_internal()
    y = frame.label_binary.astype(int).to_numpy()
    folds = frame.outer_fold.astype(int).to_numpy()
    ids = frame.public_subject_id.astype(str).to_numpy()
    embedding_index = frame.embedding_index.astype(int).to_numpy()
    require_binary(y, "internal cohort")
    predictions = []
    selection_rows = []
    provenance = []
    for outer in range(5):
        tr = np.flatnonzero(folds != outer)
        held = np.flatnonzero(folds == outer)
        if len(tr) != 300 or len(held) != 75 or set(ids[tr]) & set(ids[held]):
            raise ValueError(f"Invalid outer split {outer}")
        require_binary(y[tr], f"outer {outer} training")
        p_c_inner = clinical_inner_oof(frame.iloc[tr], y[tr], outer)
        p_b_inner = np.full(300, np.nan)
        for inner, (itr, iva) in enumerate(StratifiedKFold(5, shuffle=True, random_state=20260909 + outer).split(X[tr], y[tr])):
            model = fit_probe(X, y, tr[itr], embedding_index)
            p_b_inner[iva] = model.predict_proba(X[tr[iva]])[:, 1]
            provenance.append({
                "outer_fold": outer, "inner_fold": inner, "component": "B_image",
                "n_fit": len(itr), "n_inner_heldout": len(iva),
                "n_meta_heldout_in_fit": len(set(ids[tr[itr]]) & set(ids[held])),
                "fit_ids_sha256": hashlib.sha256("\n".join(sorted(ids[tr[itr]])).encode()).hexdigest(),
                "inner_heldout_ids_sha256": hashlib.sha256("\n".join(sorted(ids[tr[iva]])).encode()).hexdigest(),
            })
        if not np.isfinite(p_b_inner).all():
            raise ValueError(f"B inner OOF incomplete for outer {outer}")
        selection = pd.DataFrame({"label": y[tr], "p_clinical": p_c_inner, "p_image": p_b_inner})
        weight, _, grid = select_weight_from_df(selection)
        grid.insert(0, "outer_fold", outer)
        selection_rows.append(grid)
        p_c_held = clinical_heldout(frame.iloc[tr], frame.iloc[held], y[tr], outer)
        probe = fit_probe(X, y, tr, embedding_index)
        p_b_held = probe.predict_proba(X[held])[:, 1]
        predictions.append(pd.DataFrame({
            "public_subject_id": ids[held], "label": y[held], "outer_fold": outer,
            "p_clinical": p_c_held, "p_image": p_b_held,
            "weight_clinical": weight,
            "p_fusion": weight * p_c_held + (1 - weight) * p_b_held,
        }))
        provenance.append({
            "outer_fold": outer, "inner_fold": None, "component": "outer_heldout",
            "n_fit": len(tr), "n_inner_heldout": len(held), "n_meta_heldout_in_fit": 0,
            "fit_ids_sha256": hashlib.sha256("\n".join(sorted(ids[tr])).encode()).hexdigest(),
            "inner_heldout_ids_sha256": hashlib.sha256("\n".join(sorted(ids[held])).encode()).hexdigest(),
        })
        print(f"outer {outer} complete, weight={weight:.2f}", flush=True)
    oof = pd.concat(predictions, ignore_index=True).sort_values("public_subject_id")
    if len(oof) != 375 or oof.public_subject_id.nunique() != 375 or set(oof.public_subject_id) != set(ids):
        raise ValueError("Final B OOF coverage failure")
    if pd.DataFrame(provenance).n_meta_heldout_in_fit.ne(0).any():
        raise ValueError("Outer held-out patients entered an inner fit")
    final_weight, _, full_grid = select_weight_from_df(oof[["label", "p_clinical", "p_image"]].rename(columns={"label": "label"}))
    oof.to_csv(OUTPUT / "B_internal_oof.csv", index=False)
    pd.concat(selection_rows, ignore_index=True).to_csv(OUTPUT / "B_outer_weight_grid.csv", index=False)
    full_grid.to_csv(OUTPUT / "B_external_weight_grid_internal_only.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUTPUT / "B_training_provenance.csv", index=False)
    write_json(OUTPUT / "internal_weight_lock.json", {
        "status": "INTERNAL_ONLY_LOCKED_B_CANDIDATE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script_sha256": sha(Path(__file__)),
        "input_sha256": hashes,
        "meta_fold_weights": {str(k): float(v) for k, v in oof.groupby("outer_fold").weight_clinical.first().items()},
        "external_clinical_weight": float(final_weight),
        "external_image_weight": float(1 - final_weight),
        "threshold": 0.5, "positive_class": "COP=1",
        "internal_oof_sha256": sha(OUTPUT / "B_internal_oof.csv"),
        "internal_metrics": metrics(oof.label.to_numpy(), oof.p_fusion.to_numpy()),
        "external_labels_read": False,
    })
    print("Internal lock written", flush=True)

def external_phase() -> None:
    lock_file = OUTPUT / "internal_weight_lock.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    if lock.get("status") != "INTERNAL_ONLY_LOCKED_B_CANDIDATE":
        raise ValueError("Internal-only lock missing")
    if sha(OUTPUT / "B_internal_oof.csv") != lock["internal_oof_sha256"]:
        raise ValueError("Internal OOF changed after weight lock")
    if (OUTPUT / "B_external_predictions.csv").exists():
        raise FileExistsError("External evaluation already exists")
    internal, X, hashes = load_internal()
    if hashes != lock["input_sha256"]:
        raise ValueError("Input hashes changed after weight lock")
    ext = load_clinical_cohort("external")
    ext_map = pd.read_csv(XWALK, dtype=str).query("cohort == 'external'")
    ext_manifest = pd.read_csv(B_DIR / "external_embedding_manifest.csv")
    ext = ext.merge(ext_map[["public_subject_id", "external_tensor_id"]], on="public_subject_id", validate="one_to_one")
    ext = ext.merge(ext_manifest[["neutral_external_id", "embedding_index"]],
                    left_on="external_tensor_id", right_on="neutral_external_id", validate="one_to_one")
    if len(ext) != 33 or ext.public_subject_id.nunique() != 33:
        raise ValueError("Expected 33 unique external patient rows")
    ext_embeddings = np.load(B_DIR / "embeddings/external_embeddings.npy", mmap_mode="r")
    idx = ext.embedding_index.astype(int).to_numpy()
    if len(set(idx)) != 33 or idx.min() != 0 or idx.max() != 32:
        raise ValueError("External B embedding index invalid")
    X_ext = np.asarray(ext_embeddings[idx], dtype=np.float32)
    y = internal.label_binary.astype(int).to_numpy()
    features = clinical.FEATURE_SETS["full16"]
    p_c_ext = clinical.external_predictions(internal[features], ext[features], y, features,
                                           "random_forest", "balanced_bootstrap")
    folds = internal.outer_fold.astype(int).to_numpy()
    embedding_index = internal.embedding_index.astype(int).to_numpy()
    image_members = []
    for outer in range(5):
        train = folds != outer
        image_members.append(fit_probe(X, y, np.flatnonzero(train), embedding_index).predict_proba(X_ext)[:, 1])
    p_b_ext = np.mean(np.vstack(image_members), axis=0)
    w = float(lock["external_clinical_weight"])
    ext_out = pd.DataFrame({
        "public_subject_id": ext.public_subject_id, "label": ext.label_binary.astype(int),
        "p_clinical": p_c_ext, "p_image": p_b_ext,
        "weight_clinical": w, "p_fusion": w * p_c_ext + (1 - w) * p_b_ext,
    }).sort_values("public_subject_id")
    if not np.isfinite(ext_out[["p_clinical", "p_image", "p_fusion"]].to_numpy()).all():
        raise ValueError("Nonfinite external probabilities")
    ext_out.to_csv(OUTPUT / "B_external_predictions.csv", index=False)
    write_json(OUTPUT / "B_external_evaluation.json", {
        "status": "B_CANDIDATE_NOT_PAPER_CERTIFIED",
        "internal_lock_sha256": sha(lock_file),
        "external_predictions_sha256": sha(OUTPUT / "B_external_predictions.csv"),
        "external_embedding_sha256": sha(B_DIR / "embeddings/external_embeddings.npy"),
        "external_embedding_manifest_sha256": sha(B_DIR / "external_embedding_manifest.csv"),
        "external_metrics": metrics(ext_out.label.to_numpy(), ext_out.p_fusion.to_numpy()),
    })
    print("External B candidate evaluated", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("internal", "external"), required=True)
    args = parser.parse_args()
    if args.phase == "internal":
        internal_phase()
    else:
        external_phase()
