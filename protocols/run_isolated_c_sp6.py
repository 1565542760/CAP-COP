"""Rebuild SP6 meta-training predictions without outer-held-out exposure.

The clinical and image branches retain their published fit functions. This
runner never reads external labels and writes only to a new output directory.
"""

from __future__ import annotations

import argparse

import os

import hashlib

import json

import shutil

import sys

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

PROJECT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

FORMAL = PROJECT / "outputs/model_c_correct_identity_formal_20261028_1518"

PREVIOUS = Path(os.environ.get("CAP_COP_PROTOCOL_ROOT", Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(PREVIOUS))

import run_isolated_b_comparison as b

from cap_cop.pipelines import stable_multirepresentation_replay_pipeline as sp

from cap_cop.pipelines.locked_weighted_multimodal_fusion_pipeline import select_weight_from_df

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def id_digest(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()

def load_inputs():
    derived = FORMAL / "stable_runtime_regenerated_tabular/regenerated_tabular_inputs"
    sp.PATHS["lesion_features"] = derived / "locked_contract_lesion_phenotype.csv"
    sp.PATHS["d1_internal_features"] = derived / "internal_lesion_radiomics.csv"
    sp.PATHS["d1_external_features"] = derived / "external_lesion_radiomics.csv"
    base, ext_xw, domains = sp.load_base()
    tab_i, tab_e, tab_f, _ = sp.load_tabular_features(base, ext_xw)
    ci = pd.read_csv(FORMAL / "stable_fit_artifacts/internal_prefit_identity.csv")
    ce = pd.read_csv(FORMAL / "stable_fit_artifacts/external_prefit_identity.csv")
    if not base[["neutral_imaging_id", "label", "outer_fold"]].astype(str).reset_index(drop=True).equals(
        ci[["neutral_imaging_id", "label", "outer_fold"]].astype(str).reset_index(drop=True)
    ):
        raise ValueError("Current SP6 inputs differ from certified formal training row identity")
    if not ext_xw[["neutral_external_id", "neutral_patient_id"]].astype(str).reset_index(drop=True).equals(
        ce[["neutral_external_id", "neutral_patient_id"]].astype(str).reset_index(drop=True)
    ):
        raise ValueError("External SP6 identity differs from formal training")
    emb_i, emb_e, emb_mi, emb_me, emb_hashes = {}, {}, {}, {}, {}
    for family, suffix in [("FAMILY_C_GENESIS_FROZEN", "genesis"), ("FAMILY_D_TANGERINE_FROZEN", "tangerine")]:
        internal_paths = [FORMAL / "runtime_embeddings" / f"{pid}_{suffix}_embedding.npy" for pid in ci.public_subject_id]
        external_paths = [FORMAL / "runtime_embeddings" / f"{pid}_{suffix}_embedding.npy" for pid in ce.public_subject_id]
        missing = [str(p) for p in internal_paths + external_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{family}: missing {len(missing)} frozen embeddings")
        emb_i[family] = np.vstack([np.load(p).reshape(-1) for p in internal_paths]).astype(np.float32)
        emb_e[family] = np.vstack([np.load(p).reshape(-1) for p in external_paths]).astype(np.float32)
        emb_mi[family] = base.copy()
        emb_me[family] = ext_xw.copy()
        emb_hashes[family] = {
            "internal_files_digest": id_digest(f"{p.name}:{digest(p)}" for p in internal_paths),
            "external_files_digest": id_digest(f"{p.name}:{digest(p)}" for p in external_paths),
        }
    audit = sp.validate_stable_prefit(base, tab_i, tab_e, emb_mi, emb_me, emb_i, emb_e)
    if audit != {"internal_patients": 375, "external_patients": 33, "internal_identity_mismatches": 0,
                 "external_identity_mismatches": 0, "label_mismatches": 0, "fold_mismatches": 0}:
        raise ValueError(f"SP6 prefit audit failed: {audit}")
    frame, _, _ = b.load_internal()
    common = ci[["public_subject_id", "label", "outer_fold"]].merge(
        frame[["public_subject_id", "label_binary", "outer_fold"]], on="public_subject_id",
        suffixes=("_sp6", "_clinical"), validate="one_to_one")
    if len(common) != 375 or common.label.astype(int).ne(common.label_binary.astype(int)).any() or common.outer_fold_sp6.ne(common.outer_fold_clinical).any():
        raise ValueError("Clinical and SP6 patients, labels or outer folds differ")
    return base, ext_xw, domains, tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e, ci, frame, emb_hashes

def fit_sp6(base, ext_xw, domains, tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e,
            train_mask, test_mask, artifact_dir):
    hr_i, hr_f = sp.combine_tabular(tab_i["FAMILY_A_HANDCRAFTED"], tab_i["FAMILY_B_RAD_D1_FINAL13"],
                                     tab_f["FAMILY_A_HANDCRAFTED"], tab_f["FAMILY_B_RAD_D1_FINAL13"])
    hr_e = tab_e["FAMILY_A_HANDCRAFTED"][["neutral_external_id", "neutral_patient_id"] + tab_f["FAMILY_A_HANDCRAFTED"]].merge(
        tab_e["FAMILY_B_RAD_D1_FINAL13"][["neutral_patient_id"] + tab_f["FAMILY_B_RAD_D1_FINAL13"]],
        on="neutral_patient_id", how="left", validate="one_to_one", sort=False)
    fit_ids = set(tab_i["FAMILY_A_HANDCRAFTED"].loc[train_mask, "public_subject_id"])
    held_ids = set(tab_i["FAMILY_A_HANDCRAFTED"].loc[test_mask, "public_subject_id"])
    if len(fit_ids) != int(train_mask.sum()) or len(held_ids) != int(test_mask.sum()) or fit_ids & held_ids:
        raise ValueError("SP6 fit/held-out patient identity overlap")
    source = sp.fit_tabular_candidate("SP1_STABLE_HANDCRAFTED_ELASTICNET", "elasticnet", hr_i, hr_e,
                                      hr_f, train_mask, test_mask, domains, 12, "isolated", artifact_dir / "SP1")
    parts_i, parts_e, all_features = [], [], []
    source_status = {"SP1": source.status}
    if source.valid:
        chosen = source.selection.loc[source.selection.selected_after_pruning.eq(True), "raw_feature"].astype(str).tolist()
        parts_i.append(hr_i[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + chosen].copy())
        parts_e.append(hr_e[["neutral_external_id", "neutral_patient_id"] + chosen].copy())
        all_features.extend(chosen)
    for family, name in [("FAMILY_C_GENESIS_FROZEN", "GENESIS"), ("FAMILY_D_TANGERINE_FROZEN", "TANGERINE")]:
        selected = sp.fit_embedding_candidate(f"SP_SOURCE_{name}", name, "elasticnet", emb_mi[family], emb_i[family],
                                              emb_me[family], emb_e[family], train_mask, test_mask, domains, 8,
                                              "isolated", artifact_dir / name)
        source_status[name] = selected.status
        if not selected.valid:
            continue
        pc_i, pc_e, _, _ = sp.embedding_pc_frame(emb_i[family], emb_mi[family], train_mask,
                                                  emb_me[family], emb_e[family], artifact_dir / f"SP6_SOURCE_{name}")
        cols = selected.selection.loc[selected.selection.selected_after_pruning.eq(True), "raw_feature"].astype(str).tolist()
        renames = {col: f"{name}_{col}" for col in cols}
        parts_i.append(pc_i[["neutral_imaging_id", "label", "outer_fold", "primary_domain"] + cols].rename(columns=renames))
        parts_e.append(pc_e[["neutral_external_id", "neutral_patient_id"] + cols].rename(columns=renames))
        all_features.extend(renames.values())
    if not parts_i:
        raise RuntimeError(f"No valid SP6 source in this split: {source_status}")
    merged_i = parts_i[0]
    for part in parts_i[1:]:
        merged_i = merged_i.merge(part.drop(columns=["label", "outer_fold", "primary_domain"]),
                                  on="neutral_imaging_id", how="left", validate="one_to_one", sort=False)
    merged_e = parts_e[0]
    for part in parts_e[1:]:
        merged_e = merged_e.merge(part.drop(columns=["neutral_external_id"]),
                                  on="neutral_patient_id", how="left", validate="one_to_one", sort=False)
    result = sp.fit_tabular_candidate("SP6_STABLE_MULTI_REPRESENTATION_RANDOM_FOREST", "rf", merged_i, merged_e,
                                      all_features, train_mask, test_mask, domains, 20, "isolated", artifact_dir / "SP6")
    if not result.valid or len(result.pred_ids) != int(test_mask.sum()):
        raise RuntimeError(f"SP6 fit failed: {result.status}")
    expected_ids = base.loc[test_mask, "neutral_imaging_id"].astype(str).tolist()
    if result.pred_ids != expected_ids:
        raise ValueError("SP6 prediction order differs from held-out image IDs")
    return np.asarray(result.pred_p, dtype=float), {"fit_ids_sha256": id_digest(fit_ids),
                                                     "heldout_ids_sha256": id_digest(held_ids),
                                                     "n_fit": len(fit_ids), "n_heldout": len(held_ids),
                                                     "selected_dimension": result.dimension, "rf_params": result.params,
                                                     "source_status": source_status}

def run(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__), out / "run_source_snapshot.py")
    data = load_inputs()
    base, ext_xw, domains, tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e, ci, frame, emb_hashes = data
    current = pd.read_csv(FORMAL / "model_c_internal_oof_verified.csv")
    check = ci[["public_subject_id", "neutral_imaging_id", "label", "outer_fold"]].merge(
        current, on=["public_subject_id", "neutral_imaging_id", "label", "outer_fold"], validate="one_to_one")
    if len(check) != 375:
        raise ValueError("Certified SP6 OOF has missing or duplicate patients")
    source_hashes = {str(p): digest(p) for p in [
        Path(sp.__file__), Path(b.clinical.__file__), Path(__file__),
        FORMAL / "model_c_internal_oof_verified.csv",
        FORMAL / "stable_fit_artifacts/internal_prefit_identity.csv",
        FORMAL / "stable_fit_artifacts/external_prefit_identity.csv",
        *sorted((FORMAL / "stable_runtime_regenerated_tabular/regenerated_tabular_inputs").glob("*.csv")),
    ]}
    (out / "input_manifest.json").write_text(json.dumps({"source_sha256": source_hashes,
        "embedding_sha256_summary": emb_hashes, "external_labels_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    folds = [args.fold] if args.fold is not None else list(range(5))
    rows, grids, audits = [], [], []
    for outer in folds:
        tr = base.outer_fold.ne(outer).to_numpy()
        held = base.outer_fold.eq(outer).to_numpy()
        if int(tr.sum()) != 300 or int(held.sum()) != 75:
            raise ValueError(f"Invalid outer fold {outer}")
        y = base.loc[tr, "label"].astype(int).to_numpy()
        clinical_train = frame.loc[frame.outer_fold.ne(outer)].copy()
        clinical_held = frame.loc[frame.outer_fold.eq(outer)].copy()
        y_clinical = clinical_train.label_binary.astype(int).to_numpy()
        p_clinical_sorted = b.clinical_inner_oof(clinical_train, y_clinical, outer)
        p_clinical_inner = pd.Series(p_clinical_sorted, index=clinical_train.public_subject_id).loc[
            ci.loc[tr, "public_subject_id"]].to_numpy()
        p_image_inner = np.full(300, np.nan)
        positions = np.flatnonzero(tr)
        for inner, (itr, iva) in enumerate(StratifiedKFold(5, shuffle=True, random_state=20260909 + outer).split(
                np.zeros(len(y)), y)):
            inner_tr = np.zeros(375, dtype=bool)
            inner_va = np.zeros(375, dtype=bool)
            inner_tr[positions[itr]] = True
            inner_va[positions[iva]] = True
            if np.any(inner_tr & held) or np.any(inner_va & held):
                raise ValueError("Meta outer held-out entered SP6 inner split")
            p, audit = fit_sp6(base, ext_xw, domains, tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e,
                                inner_tr, inner_va, out / f"outer_{outer}" / f"inner_{inner}")
            p_image_inner[iva] = p
            audits.append({"outer_fold": outer, "inner_fold": inner, "component": "SP6", **audit,
                           "outer_heldout_in_fit": 0})
            print(f"outer={outer} inner={inner} SP6 complete", flush=True)
        if not np.isfinite(p_image_inner).all():
            raise ValueError(f"SP6 inner OOF incomplete in outer {outer}")
        training_side = pd.DataFrame({"label": y, "p_clinical": p_clinical_inner, "p_image": p_image_inner})
        training_side.insert(0, "public_subject_id", ci.loc[tr, "public_subject_id"].to_numpy())
        training_side.to_csv(out / f"outer_{outer}" / "meta_training_oof.csv", index=False)
        weight, _, grid = select_weight_from_df(training_side[["label", "p_clinical", "p_image"]])
        grid.insert(0, "outer_fold", outer)
        grids.append(grid)
        p_clinical_held_sorted = b.clinical_heldout(clinical_train, clinical_held, y_clinical, outer)
        p_clinical_held = pd.Series(p_clinical_held_sorted, index=clinical_held.public_subject_id).loc[
            ci.loc[held, "public_subject_id"]].to_numpy()
        p_image_held, audit = fit_sp6(base, ext_xw, domains, tab_i, tab_e, tab_f, emb_mi, emb_me, emb_i, emb_e,
                                      tr, held, out / f"outer_{outer}" / "heldout_refit")
        ref = current.set_index("neutral_imaging_id").loc[base.loc[held, "neutral_imaging_id"], "prob_COP"].to_numpy()
        max_delta = float(np.max(np.abs(p_image_held - ref)))
        if max_delta > 1e-7:
            raise ValueError(f"Outer {outer}: SP6 refit differs from certified fit by {max_delta}")
        audits.append({"outer_fold": outer, "inner_fold": None, "component": "SP6", **audit,
                       "outer_heldout_in_fit": 0, "certified_probability_max_delta": max_delta})
        rows.append(pd.DataFrame({"public_subject_id": ci.loc[held, "public_subject_id"].to_numpy(),
            "label": base.loc[held, "label"].astype(int).to_numpy(), "outer_fold": outer,
            "p_clinical": p_clinical_held, "p_image": p_image_held, "weight_clinical": weight,
            "p_fusion": weight * p_clinical_held + (1 - weight) * p_image_held}))
        pd.concat(rows, ignore_index=True).to_csv(out / "partial_internal_oof.csv", index=False)
        pd.concat(grids, ignore_index=True).to_csv(out / "partial_weight_grid.csv", index=False)
        pd.DataFrame(audits).to_csv(out / "partial_fit_provenance.csv", index=False)
        print(f"outer={outer} complete; weight={weight:.2f}; certified max delta={max_delta:.3g}", flush=True)
    if args.fold is not None:
        print("Pilot complete; no full-cohort weight lock or external evaluation", flush=True)
        return
    oof = pd.concat(rows, ignore_index=True).sort_values("public_subject_id")
    if len(oof) != 375 or oof.public_subject_id.nunique() != 375:
        raise ValueError("SP6 strict OOF does not cover 375 unique patients")
    w, _, full_grid = select_weight_from_df(oof[["label", "p_clinical", "p_image"]])
    oof.to_csv(out / "C_strict_internal_oof.csv", index=False)
    full_grid.to_csv(out / "C_external_weight_grid_internal_only.csv", index=False)
    lock = {"status": "INTERNAL_ONLY_LOCKED_C_CANDIDATE", "external_labels_read": False,
            "external_clinical_weight": float(w), "internal_oof_sha256": digest(out / "C_strict_internal_oof.csv"),
            "fold_weights": {str(k): float(v) for k, v in oof.groupby("outer_fold").weight_clinical.first().items()},
            "internal_metrics": b.metrics(oof.label.to_numpy(), oof.p_fusion.to_numpy())}
    (out / "internal_weight_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Internal-only lock complete; external clinical weight={w:.2f}", flush=True)

def external_phase(output: str):
    out = Path(output).resolve()
    lock_path = out / "internal_weight_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "INTERNAL_ONLY_LOCKED_C_CANDIDATE" or lock.get("external_labels_read") is not False:
        raise ValueError("Valid internal-only C lock is required before external evaluation")
    internal_path = out / "C_strict_internal_oof.csv"
    if digest(internal_path) != lock["internal_oof_sha256"]:
        raise ValueError("C internal OOF changed after weight lock")
    if (out / "C_strict_external_predictions.csv").exists():
        raise FileExistsError("External C evaluation already exists")
    internal, _, _ = b.load_internal()
    external = b.load_clinical_cohort("external")
    image = pd.read_csv(FORMAL / "model_c_external_mean_verified.csv")
    if len(external) != 33 or external.public_subject_id.nunique() != 33 or len(image) != 33:
        raise ValueError("External clinical or corrected SP6 patient coverage invalid")
    merged = external.merge(image[["public_subject_id", "prob_COP", "label"]], on="public_subject_id",
                            validate="one_to_one")
    if len(merged) != 33 or merged.label_binary.astype(int).ne(merged.label.astype(int)).any():
        raise ValueError("Clinical and corrected SP6 external patient labels differ")
    features = b.clinical.FEATURE_SETS["full16"]
    p_clinical = b.clinical.external_predictions(
        internal[features], merged[features], internal.label_binary.astype(int).to_numpy(),
        features, "random_forest", "balanced_bootstrap")
    ref = pd.read_csv(PROJECT / "outputs/model_c_correct_identity_weight_selection_20261028/external_frozen_new_weight.csv")
    check = merged[["public_subject_id"]].assign(p_clinical=p_clinical).merge(
        ref[["public_subject_id", "p_clinical"]], on="public_subject_id", validate="one_to_one", suffixes=("_new", "_ref"))
    clinical_delta = float(np.max(np.abs(check.p_clinical_new - check.p_clinical_ref)))
    if clinical_delta > 1e-7:
        raise ValueError(f"Clinical external refit differs from verified reference by {clinical_delta}")
    w = float(lock["external_clinical_weight"])
    result = pd.DataFrame({"public_subject_id": merged.public_subject_id, "label": merged.label.astype(int),
        "p_clinical": p_clinical, "p_image": merged.prob_COP.astype(float), "weight_clinical": w,
        "p_fusion": w * p_clinical + (1 - w) * merged.prob_COP.astype(float).to_numpy()})
    result = result.sort_values("public_subject_id")
    if not np.isfinite(result[["p_clinical", "p_image", "p_fusion"]].to_numpy()).all():
        raise ValueError("Nonfinite C external probabilities")
    path = out / "C_strict_external_predictions.csv"
    result.to_csv(path, index=False)
    summary = {"status": "C_STRICT_META_EXTERNAL_EVALUATED", "internal_lock_sha256": digest(lock_path),
        "external_prediction_sha256": digest(path), "external_image_source_sha256": digest(FORMAL / "model_c_external_mean_verified.csv"),
        "external_clinical_reference_max_delta": clinical_delta,
        "metrics": b.metrics(result.label.to_numpy(), result.p_fusion.to_numpy())}
    (out / "C_external_evaluation.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--phase", choices=("internal", "external"), default="internal")
    args = parser.parse_args()
    if args.phase == "internal":
        run(args)
    else:
        external_phase(args.output)
