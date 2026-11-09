"""Lock strict A internal weights, then evaluate external patients separately."""

from __future__ import annotations

import os

import argparse

import hashlib

import json

import sys

from datetime import datetime, timezone

from pathlib import Path

import numpy as np

import pandas as pd

PROJECT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

PREVIOUS = Path(os.environ.get("CAP_COP_PROTOCOL_ROOT", Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(PREVIOUS))

import run_isolated_b_comparison as b

FOLDS = [PROJECT / "outputs" / ("strict_meta_a_sklearn161_20261029_outer_1_v2" if fold == 1
                            else f"strict_meta_a_sklearn161_20261029_outer_{fold}") for fold in range(5)]

REF = PROJECT / "models/lesion_radiomics/internal_oof_predictions.csv"

FINAL_REFIT = PROJECT / "outputs/strict_meta_a_final_refit_sklearn161_20261029_01"

EXT_IMAGE = FINAL_REFIT / "A_external_label_blind_refit_predictions.csv"

XWALK = PROJECT / "data/public_subject_model_crosswalk.csv"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def internal(out: Path):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite A result: {out}")
    out.mkdir(parents=True)
    rows, grids, hashes = [], [], {}
    for fold, root in enumerate(FOLDS):
        source_manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
        if source_manifest.get("environment", {}).get("scikit_learn") != "1.6.1":
            raise ValueError(f"A fold {fold} was not fitted in the locked 1.6.1 environment")
        pred_path = root / "outer_heldout_predictions.csv"
        pred = pd.read_csv(pred_path)
        if len(pred) != 75 or pred.public_subject_id.nunique() != 75 or pred.outer_fold.ne(fold).any():
            raise ValueError(f"A outer fold {fold} coverage invalid")
        result = json.loads((root / "outer_result.json").read_text(encoding="utf-8"))
        delta = result.get("historical_image_max_delta", result.get("certified_image_max_delta"))
        if delta is None or not np.isfinite(delta) or not np.allclose(pred.p_fusion,
                pred.weight_clinical * pred.p_clinical + (1 - pred.weight_clinical) * pred.p_image):
            raise ValueError(f"A outer fold {fold} source or formula invalid")
        if pred.weight_clinical.nunique() != 1 or abs(pred.weight_clinical.iloc[0] - result["weight_clinical"]) > 1e-12:
            raise ValueError(f"A outer fold {fold} weight differs from fit record")
        for inner in range(5):
            root_fit = root / f"meta_inner_{inner}"
            if not (root_fit / "model.joblib").is_file() or not (root_fit / "fit_manifest.json").is_file():
                raise ValueError(f"A inner model missing: outer {fold}, inner {inner}")
            fit = json.loads((root_fit / "fit_manifest.json").read_text(encoding="utf-8"))
            if sha(root_fit / "model.joblib") != fit["model_sha256"]:
                raise ValueError(f"A inner model hash mismatch: outer {fold}, inner {inner}")
        rows.append(pred)
        grid = pd.read_csv(root / "weight_grid.csv")
        grid.insert(0, "outer_fold", fold)
        grids.append(grid)
        hashes[str(root)] = {"prediction_sha256": sha(pred_path),
                             "source_sha256": sha(root / "run_source_snapshot.py"),
                             "weight_grid_sha256": sha(root / "weight_grid.csv")}
    merged = pd.concat(rows, ignore_index=True).sort_values("public_subject_id")
    xwalk = pd.read_csv(XWALK)
    truth = xwalk.loc[xwalk.cohort.eq("internal"), ["public_subject_id", "label_binary", "segmentation_model_id"]]
    verified = merged.merge(truth, on="public_subject_id", validate="one_to_one")
    if len(verified) != 375 or verified.public_subject_id.nunique() != 375 or verified.label.ne(verified.label_binary).any() or verified.neutral_patient_id.ne(verified.segmentation_model_id).any():
        raise ValueError("A final OOF patient, label or lesion ID mismatch")
    reference = pd.read_csv(REF)
    checked = merged.merge(reference[["neutral_patient_id", "probability_COP", "outer_fold"]],
                           on="neutral_patient_id", validate="one_to_one", suffixes=("", "_ref"))
    image_delta = float(np.max(np.abs(checked.p_image - checked.probability_COP)))
    if checked.outer_fold.ne(checked.outer_fold_ref).any():
        raise ValueError("A image refit and historical reference have different outer folds")
    b_oof = pd.read_csv(PROJECT / "outputs/formal_comparison_meta_isolation_20261028_b_v3/B_internal_oof.csv")
    shared = merged.merge(b_oof[["public_subject_id", "p_clinical"]], on="public_subject_id",
                          validate="one_to_one", suffixes=("", "_b"))
    clinical_delta = float(np.max(np.abs(shared.p_clinical - shared.p_clinical_b)))
    if clinical_delta > 1e-10:
        raise ValueError(f"A clinical OOF differs from common B/C branch by {clinical_delta}")
    weight, _, full_grid = b.select_weight_from_df(merged[["label", "p_clinical", "p_image"]])
    pred_path = out / "A_strict_internal_oof.csv"
    merged.to_csv(pred_path, index=False)
    pd.concat(grids, ignore_index=True).to_csv(out / "A_outer_weight_grid.csv", index=False)
    full_grid.to_csv(out / "A_external_weight_grid_internal_only.csv", index=False)
    manifest = {"status": "INTERNAL_ONLY_LOCKED_A_CANDIDATE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "external_labels_read": False, "source_script_sha256": sha(Path(__file__)),
                "input_sha256": {str(REF): sha(REF), str(XWALK): sha(XWALK), **hashes},
                "internal_oof_sha256": sha(pred_path), "historical_image_max_delta": image_delta,
                "image_branch_status": "CURRENT_CODE_REFIT_HISTORICAL_DELTA_RECORDED",
                "shared_clinical_max_delta": clinical_delta,
                "fold_weights": {str(k): float(v) for k, v in merged.groupby("outer_fold").weight_clinical.first().items()},
                "external_clinical_weight": float(weight), "internal_metrics": b.metrics(merged.label.to_numpy(), merged.p_fusion.to_numpy())}
    (out / "internal_weight_lock.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "AUROC": manifest["internal_metrics"]["AUROC"],
                      "weights": manifest["fold_weights"], "external_weight": weight}), flush=True)

def external(out: Path):
    lock_path = out / "internal_weight_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    internal_path = out / "A_strict_internal_oof.csv"
    if lock.get("status") != "INTERNAL_ONLY_LOCKED_A_CANDIDATE" or lock.get("external_labels_read") is not False or sha(internal_path) != lock["internal_oof_sha256"]:
        raise ValueError("A internal-only lock invalid or prediction changed")
    if (out / "A_strict_external_predictions.csv").exists():
        raise FileExistsError("A external evaluation already exists")
    refit = json.loads((FINAL_REFIT / "run_manifest.json").read_text(encoding="utf-8"))
    if refit.get("status") != "A_FINAL_MODEL_REFIT_AND_EXTERNAL_LABEL_BLIND_PREDICTED" or refit.get("external_labels_read") is not False:
        raise ValueError("A new final image model was not locked before external labels")
    if refit.get("environment", {}).get("scikit_learn") != "1.6.1":
        raise ValueError("A final image model was not fitted in the locked 1.6.1 environment")
    if sha(EXT_IMAGE) != refit["external_predictions_sha256"] or sha(FINAL_REFIT / "final/lesion_radiomics_replay_FINAL_MODEL.joblib") != refit["final_model_sha256"]:
        raise ValueError("A new final image model or predictions changed after lock")
    internal_clinical, _, _ = b.load_internal()
    external_clinical = b.load_clinical_cohort("external")
    xwalk = pd.read_csv(XWALK, dtype=str).query("cohort == 'external'")
    image = pd.read_csv(EXT_IMAGE)
    merged = external_clinical.merge(xwalk[["public_subject_id", "segmentation_model_id", "label_binary"]],
                                     on="public_subject_id", validate="one_to_one", suffixes=("", "_map"))
    merged = merged.merge(image[["neutral_patient_id", "probability_COP"]],
                          left_on="segmentation_model_id", right_on="neutral_patient_id", validate="one_to_one")
    if len(merged) != 33 or merged.public_subject_id.nunique() != 33 or merged.label_binary.astype(int).ne(merged.label_binary_map.astype(int)).any():
        raise ValueError("A external identity or label mismatch")
    features = b.clinical.FEATURE_SETS["full16"]
    p_clinical = b.clinical.external_predictions(internal_clinical[features], merged[features],
        internal_clinical.label_binary.astype(int).to_numpy(), features, "random_forest", "balanced_bootstrap")
    c_ext = pd.read_csv(PROJECT / "outputs/strict_meta_c_sp6_full_20261029_02/C_strict_external_predictions.csv")
    check = pd.DataFrame({"public_subject_id": merged.public_subject_id, "p_clinical": p_clinical}).merge(
        c_ext[["public_subject_id", "p_clinical"]], on="public_subject_id", validate="one_to_one", suffixes=("", "_c"))
    clinical_delta = float(np.max(np.abs(check.p_clinical - check.p_clinical_c)))
    if clinical_delta > 1e-10:
        raise ValueError(f"A external clinical differs from shared C source by {clinical_delta}")
    w = float(lock["external_clinical_weight"])
    result = pd.DataFrame({"public_subject_id": merged.public_subject_id, "label": merged.label_binary.astype(int),
        "p_clinical": p_clinical, "p_image": merged.probability_COP.astype(float), "weight_clinical": w,
        "p_fusion": w * p_clinical + (1 - w) * merged.probability_COP.astype(float).to_numpy()}).sort_values("public_subject_id")
    if not np.isfinite(result[["p_clinical", "p_image", "p_fusion"]].to_numpy()).all():
        raise ValueError("A external nonfinite prediction")
    pred_path = out / "A_strict_external_predictions.csv"
    result.to_csv(pred_path, index=False)
    report = {"status": "A_STRICT_META_EXTERNAL_EVALUATED", "internal_lock_sha256": sha(lock_path),
              "external_prediction_sha256": sha(pred_path), "external_image_source_sha256": sha(EXT_IMAGE),
              "final_image_model_sha256": refit["final_model_sha256"],
              "shared_clinical_max_delta": clinical_delta,
              "metrics": b.metrics(result.label.to_numpy(), result.p_fusion.to_numpy())}
    (out / "A_external_evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("internal", "external"), required=True)
    args = parser.parse_args()
    (internal if args.phase == "internal" else external)(Path(args.output).resolve())
