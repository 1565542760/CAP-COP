# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import hashlib

import importlib.util

import json

import math

import os

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import joblib

from scipy.stats import spearmanr

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import RobustScaler, StandardScaler

from cap_cop.data.model_contracts import (
    align_stable_patient_features,
    read_stable_cohort_metadata,
    require_unique_ids,
    stable_patient_contract,
)

PROJECT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[3]))

MODEL_DIR = PROJECT / "models" / "stable_multirepresentation_ct"

WHOLE_CHEST_MODEL_DIR = PROJECT / "models" / "whole_chest_ct"

ROOT = PROJECT / "outputs" / "stable_multirepresentation_pipeline"

CODE = PROJECT / "src" / "cap_cop"

SEED = 20260909

INNER_SPLITS = 3

EPS = 1e-8

PATHS = {
    "internal_oof_ref": MODEL_DIR / "derived_inputs" / "09_V3_4_STANDARD_INTERNAL_OOF_PREDICTIONS.csv",
    "lodo_ref": MODEL_DIR / "derived_inputs" / "09_V3_4_STANDARD_INTERNAL_OOF_PREDICTIONS.csv",
    "domain_lock": MODEL_DIR / "derived_inputs" / "03_PRIMARY_DG_DOMAIN_LOCK.json",
    "v34_internal_metrics": MODEL_DIR / "internal_oof_predictions.csv",
    "v34_lodo_metrics": MODEL_DIR / "internal_oof_predictions.csv",
    "v34_external_metrics": MODEL_DIR / "external_predictions.csv",
    "v35_internal_metrics": WHOLE_CHEST_MODEL_DIR / "internal_oof_predictions.csv",
    "v35_lodo_metrics": WHOLE_CHEST_MODEL_DIR / "internal_oof_predictions.csv",
    "v35_external_metrics": WHOLE_CHEST_MODEL_DIR / "external_predictions.csv",
    "internal_crosswalk": PROJECT / "data" / "public_subject_model_crosswalk.csv",
    "external_crosswalk": MODEL_DIR / "derived_inputs" / "EXTERNAL_EXTV2_TO_EXTERNAL_PATIENT_CROSSWALK.csv",
    "external_labels": MODEL_DIR / "derived_inputs" / "EXTERNAL_LOCKED_PREDICTIONS_WITH_LABELS_V2.csv",
    "lesion_features": MODEL_DIR / "derived_inputs" / "06_LESION_PHENOTYPE_FEATURES.csv",
    "v37_feature_registry": MODEL_DIR / "derived_inputs" / "06_LESION_PHENOTYPE_FEATURES.csv",
    "d1_internal_features": MODEL_DIR / "derived_inputs" / "V3_5_D1_INTERNAL375_BASELINE_PYRADIOMICS_FEATURES.csv",
    "d1_external_features": MODEL_DIR / "derived_inputs" / "V3_5_D1_EXTERNAL33_FEATURES_LABEL_BLIND.csv",
    "d1_feature_list": MODEL_DIR / "derived_inputs" / "V3_5_D1_FINAL_FEATURE_LIST.csv",
    "d1_internal_metrics": MODEL_DIR / "internal_oof_predictions.csv",
    "d1_external_metrics": MODEL_DIR / "external_predictions.csv",
    "tangerine_internal_embeddings": WHOLE_CHEST_MODEL_DIR / "embeddings" / "internal_embeddings.npy",
    "tangerine_internal_manifest": WHOLE_CHEST_MODEL_DIR / "internal_embedding_manifest.csv",
    "tangerine_external_embeddings": WHOLE_CHEST_MODEL_DIR / "embeddings" / "external_embeddings.npy",
    "tangerine_external_manifest": WHOLE_CHEST_MODEL_DIR / "external_embedding_manifest.csv",
    "genesis_checkpoint": MODEL_DIR / "foundation_encoders" / "genesis_chest_ct.pt",
    "genesis_code": Path(__file__).resolve().parents[1] / "vendor" / "models_genesis" / "unet3d.py",
    "internal_tensor_dir": PROJECT / "outputs" / "runtime_intermediate" / "internal_p95_tensors",
    "external_tensor_dir": PROJECT / "outputs" / "runtime_intermediate" / "external_p95_tensors",
}

DIRS = {
    "protocol": ROOT / "00_protocol",
    "literature": ROOT / "01_literature",
    "audit": ROOT / "02_feature_audit",
    "handcrafted": ROOT / "03_handcrafted",
    "genesis": ROOT / "04_genesis_embedding",
    "tangerine": ROOT / "05_tangerine_embedding",
    "selection": ROOT / "06_stability_selection",
    "oof": ROOT / "07_internal_oof",
    "lodo": ROOT / "08_lodo",
    "models": ROOT / "09_models",
    "external": ROOT / "10_external_predictions",
    "metrics": ROOT / "11_metrics",
    "reports": ROOT / "12_reports",
    "locks": ROOT / "13_locks",
    "logs": ROOT / "14_logs",
    "code": CODE,
}

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def record_candidate_inputs(artifact_dir: Path | None, train: pd.DataFrame,
                            test: pd.DataFrame, external: pd.DataFrame,
                            xtr: np.ndarray, xte: np.ndarray, xext: np.ndarray) -> None:
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in (("train", train), ("held_out", test), ("external", external)):
        frame.to_csv(artifact_dir / f"{name}_selected_raw.csv", index=False)
    for name, array in (("train", xtr), ("held_out", xte), ("external", xext)):
        np.save(artifact_dir / f"{name}_model_matrix.npy", array, allow_pickle=False)

def record_fitted_candidate(artifact_dir: Path | None, train: pd.DataFrame,
                            test: pd.DataFrame, external: pd.DataFrame,
                            selected: list[str], xtr: np.ndarray, xte: np.ndarray,
                            xext: np.ndarray, imputer: SimpleImputer,
                            scaler: RobustScaler, model: Any,
                            params: dict[str, Any]) -> None:
    if artifact_dir is None:
        return
    if not artifact_dir.exists():
        record_candidate_inputs(artifact_dir, train, test, external, xtr, xte, xext)
    if (artifact_dir / "artifact_manifest.json").exists():
        raise FileExistsError("Fitted candidate artifact already exists")
    for name, value in (("imputer", imputer), ("scaler", scaler), ("model", model)):
        joblib.dump(value, artifact_dir / f"{name}.joblib")
    payload = {
        "selected_features": selected,
        "selected_params": params,
        "rows": {"train": len(train), "held_out": len(test), "external": len(external)},
        "files_sha256": {p.name: sha256_file(p) for p in sorted(artifact_dir.iterdir()) if p.is_file()},
    }
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan, "LogLoss": np.nan}
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
    }

def hedges_g(cap: np.ndarray, cop: np.ndarray) -> tuple[float, float]:
    cap = np.asarray(cap, dtype=float)
    cop = np.asarray(cop, dtype=float)
    cap = cap[np.isfinite(cap)]
    cop = cop[np.isfinite(cop)]
    n0, n1 = len(cap), len(cop)
    if n0 < 2 or n1 < 2:
        return np.nan, np.nan
    s0, s1 = np.var(cap, ddof=1), np.var(cop, ddof=1)
    sp = math.sqrt(((n0 - 1) * s0 + (n1 - 1) * s1) / max(n0 + n1 - 2, 1))
    if sp <= EPS:
        return np.nan, np.nan
    d = (np.mean(cop) - np.mean(cap)) / sp
    j = 1 - 3 / max(4 * (n0 + n1) - 9, 1)
    g = float(j * d)
    var_g = float((n0 + n1) / (n0 * n1) + (g * g) / (2 * max(n0 + n1 - 2, 1)))
    return g, var_g

def random_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    yi = np.array([r["domain_g"] for r in rows], dtype=float)
    vi = np.array([r["domain_var"] for r in rows], dtype=float)
    ok = np.isfinite(yi) & np.isfinite(vi) & (vi > 0)
    yi, vi = yi[ok], vi[ok]
    k = len(yi)
    if k == 0:
        return {"pooled_g": np.nan, "ci_low": np.nan, "ci_high": np.nan, "tau2": np.nan, "I2": np.nan}
    if k == 1:
        se = math.sqrt(float(vi[0]))
        return {"pooled_g": float(yi[0]), "ci_low": float(yi[0] - 1.96 * se), "ci_high": float(yi[0] + 1.96 * se), "tau2": 0.0, "I2": 0.0}
    w = 1 / vi
    fixed = float(np.sum(w * yi) / np.sum(w))
    q = float(np.sum(w * (yi - fixed) ** 2))
    df = k - 1
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (vi + tau2)
    pooled = float(np.sum(wr * yi) / np.sum(wr))
    se = math.sqrt(float(1 / np.sum(wr)))
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {"pooled_g": pooled, "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se, "tau2": tau2, "I2": float(i2)}

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_base() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metadata = PATHS.get("cohort_metadata")
    base = read_stable_cohort_metadata(
        metadata if metadata is not None else PATHS["internal_oof_ref"],
        legacy=metadata is None,
    )
    with PATHS["domain_lock"].open("r", encoding="utf-8") as f:
        lock = json.load(f)
    domains = list(lock["lodo_evaluable_domains"])
    ext_xw = read_csv(PATHS["external_crosswalk"])
    return base, ext_xw, domains

def numeric_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
            df[c] = s
    return cols

def load_tabular_features(base: pd.DataFrame, ext_xw: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[str]], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    internal: dict[str, pd.DataFrame] = {}
    external: dict[str, pd.DataFrame] = {}
    feats: dict[str, list[str]] = {}

    crosswalk = read_csv(PATHS["internal_crosswalk"])
    base_identity, external_identity = stable_patient_contract(base, ext_xw, crosswalk)
    lesion = read_csv(PATHS["lesion_features"])
    a_feats = numeric_cols(lesion, {"public_subject_id", "neutral_patient_id", "source_neutral_imaging_id",
                                    "neutral_external_id", "site", "group", "label_binary", "outer_fold"})
    li = align_stable_patient_features(base_identity, lesion.loc[lesion["site"].eq("Internal")].copy(),
                                       a_feats, cohort="internal", check_lesion_id=True)
    external_lesion = lesion.loc[lesion["site"].eq("External")].copy()
    external_lesion_source = external_lesion["source_neutral_imaging_id"].fillna("").astype(str).str.strip().ne("")
    if external_lesion_source.any() and not external_lesion_source.all():
        raise ValueError("External phenotype has a partially missing source imaging ID")
    le = align_stable_patient_features(external_identity, external_lesion, a_feats,
                                       cohort="external", check_external_tensor=True,
                                       check_lesion_id=bool(external_lesion_source.all()))
    internal["FAMILY_A_HANDCRAFTED"] = li[["public_subject_id", "neutral_imaging_id", "label", "outer_fold", "primary_domain"] + a_feats]
    external["FAMILY_A_HANDCRAFTED"] = le[["public_subject_id", "neutral_external_id", "neutral_patient_id"] + a_feats]
    feats["FAMILY_A_HANDCRAFTED"] = a_feats
    audit.append({"resource": "FAMILY_A_HANDCRAFTED", "path": str(PATHS["lesion_features"]), "available": True, "internal_rows": len(li), "external_rows": len(le), "feature_count": len(a_feats), "status": "EXECUTABLE"})

    d1i = read_csv(PATHS["d1_internal_features"])
    d1e = read_csv(PATHS["d1_external_features"])
    d1_list = read_csv(PATHS["d1_feature_list"])["feature"].astype(str).tolist()
    if len(d1_list) != len(set(d1_list)) or set(d1_list).difference(d1i) or set(d1_list).difference(d1e):
        raise ValueError("Locked radiomics feature list is duplicated or absent from an input")
    d1_feats = d1_list
    d1i = align_stable_patient_features(base_identity, d1i, d1_feats,
                                        cohort="internal radiomics", check_lesion_id=True)
    d1e_has_source = "source_neutral_imaging_id" in d1e.columns
    if d1e_has_source:
        source_present = d1e["source_neutral_imaging_id"].fillna("").astype(str).str.strip().ne("")
        if source_present.any() and not source_present.all():
            raise ValueError("External radiomics has a partially missing source imaging ID")
        d1e_has_source = bool(source_present.all())
    d1e = align_stable_patient_features(external_identity, d1e, d1_feats,
                                        cohort="external radiomics", check_lesion_id=d1e_has_source)
    for f in d1_feats:
        d1i[f] = pd.to_numeric(d1i[f], errors="coerce")
        d1e[f] = pd.to_numeric(d1e[f], errors="coerce")
    internal["FAMILY_B_RAD_D1_FINAL13"] = d1i[["public_subject_id", "neutral_imaging_id", "label", "outer_fold", "primary_domain"] + d1_feats]
    external["FAMILY_B_RAD_D1_FINAL13"] = d1e[["public_subject_id", "neutral_external_id", "neutral_patient_id"] + d1_feats]
    feats["FAMILY_B_RAD_D1_FINAL13"] = d1_feats
    audit.append({"resource": "FAMILY_B_RAD_D1_FINAL13", "path": str(PATHS["d1_internal_features"]), "available": True, "internal_rows": len(d1i), "external_rows": len(d1e), "feature_count": len(d1_feats), "status": "EXECUTABLE"})
    return internal, external, feats, audit

def import_genesis():
    spec = importlib.util.spec_from_file_location("v38_models_genesis_unet3d", PATHS["genesis_code"])
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def extract_genesis_embeddings(ids: list[str], tensor_dir: Path, out_npy: Path, out_manifest: Path) -> bool:
    if out_npy.exists() and out_manifest.exists():
        return True
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        write_text(DIRS["genesis"] / "GENESIS_EXTRACTION_ERROR.txt", f"torch import failed: {exc}\n")
        return False
    try:
        mod = import_genesis()
        raw = torch.load(PATHS["genesis_checkpoint"], map_location="cpu", weights_only=False)
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        state = {str(k).replace("module.", "").replace("model.", "").replace("encoder.", ""): v for k, v in state.items() if torch.is_tensor(v)}
        model = mod.UNet3D(n_class=1, act="relu")
        keep = {k: v for k, v in state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        pool = nn.AdaptiveAvgPool3d(1)
        rows, manifest = [], []
        with torch.no_grad():
            for i, nid in enumerate(ids):
                arr = np.load(tensor_dir / f"{nid}.npy").astype(np.float32, copy=False)
                if arr.ndim == 3:
                    arr = arr[None, None]
                elif arr.ndim == 4:
                    arr = arr[None]
                x = torch.from_numpy(arr).to(device)
                out64, _ = model.down_tr64(x)
                out128, _ = model.down_tr128(out64)
                out256, _ = model.down_tr256(out128)
                out512, _ = model.down_tr512(out256)
                emb = pool(out512).flatten(1).detach().cpu().numpy()[0].astype(np.float32)
                rows.append(emb)
                manifest.append({"id": nid, "embedding_index": i, "embedding_dim": int(emb.shape[0]), "source_tensor": str(tensor_dir / f"{nid}.npy")})
        np.save(out_npy, np.vstack(rows).astype(np.float32))
        pd.DataFrame(manifest).to_csv(out_manifest, index=False)
        return True
    except Exception as exc:
        write_text(DIRS["genesis"] / "GENESIS_EXTRACTION_ERROR.txt", f"{type(exc).__name__}: {exc}\n")
        return False

def meta_select(df: pd.DataFrame, features: list[str], train_mask: np.ndarray, domains: list[str], max_features: int, prefix: str = "") -> tuple[list[str], pd.DataFrame]:
    train = df.loc[train_mask].copy()
    rows: list[dict[str, Any]] = []
    for f in features:
        vals = pd.to_numeric(train[f], errors="coerce")
        if vals.isna().mean() > 0.20:
            rows.append({"feature": f, "selection_tier": "EXCLUDED_MISSING_GT20", "selected": False})
            continue
        dom_rows = []
        for d in domains:
            sub = train[train["primary_domain"].eq(d)]
            cap = pd.to_numeric(sub.loc[sub["label"].eq(0), f], errors="coerce").dropna().to_numpy(float)
            cop = pd.to_numeric(sub.loc[sub["label"].eq(1), f], errors="coerce").dropna().to_numpy(float)
            if len(cap) < 5 or len(cop) < 5:
                continue
            g, var = hedges_g(cap, cop)
            if np.isfinite(g) and np.isfinite(var):
                dom_rows.append({"domain": d, "domain_g": g, "domain_var": var, "n_CAP": len(cap), "n_COP": len(cop)})
        meta = random_effects(dom_rows)
        k = len(dom_rows)
        pooled = meta["pooled_g"]
        direction = np.sign(pooled) if np.isfinite(pooled) and pooled != 0 else 0
        dir_cons = float(np.mean([np.sign(r["domain_g"]) == direction for r in dom_rows])) if k and direction else 0.0
        primary = k >= 3 and abs(pooled) >= 0.30 and dir_cons >= 0.75 and meta["I2"] <= 60
        fallback = k >= 3 and abs(pooled) >= 0.20 and dir_cons >= 0.67 and meta["I2"] <= 75
        rows.append({
            "feature": prefix + f,
            "raw_feature": f,
            "evaluable_domains": k,
            "pooled_Hedges_g": pooled,
            "ci_low": meta["ci_low"],
            "ci_high": meta["ci_high"],
            "tau2": meta["tau2"],
            "I2": meta["I2"],
            "direction_consistency": dir_cons,
            "selection_tier": "PRIMARY" if primary else ("FALLBACK_ELIGIBLE" if fallback else "NOT_STABLE"),
            "selected": primary,
        })
    res = pd.DataFrame(rows)
    selected = res.loc[res["selected"].eq(True), "raw_feature"].astype(str).tolist() if not res.empty else []
    tier = "PRIMARY"
    if len(selected) < 3:
        res["selected"] = res["selection_tier"].isin(["PRIMARY", "FALLBACK_ELIGIBLE"])
        selected = res.loc[res["selected"].eq(True), "raw_feature"].astype(str).tolist()
        tier = "FALLBACK"
    if len(selected) < 3:
        res["selected"] = False
        res["selection_tier_final"] = "INSUFFICIENT_STABLE_FEATURES"
        return [], res
    res["selection_tier_final"] = tier
    selected = prune_features(train, selected, res, max_features)
    res["selected_after_pruning"] = res["raw_feature"].isin(selected)
    return selected, res

def prune_features(train: pd.DataFrame, selected: list[str], meta: pd.DataFrame, max_features: int) -> list[str]:
    selected = list(dict.fromkeys(selected))
    score = meta.set_index("raw_feature")
    keep: list[str] = []
    for f in sorted(selected, key=lambda x: (-abs(float(score.loc[x, "pooled_Hedges_g"])), float(score.loc[x, "I2"]), x)):
        redundant = False
        for k in keep:
            a = pd.to_numeric(train[f], errors="coerce")
            b = pd.to_numeric(train[k], errors="coerce")
            ok = a.notna() & b.notna()
            rho = spearmanr(a[ok], b[ok]).correlation if ok.sum() > 3 else 0
            if np.isfinite(rho) and abs(rho) > 0.85:
                redundant = True
                break
        if not redundant:
            keep.append(f)
        if len(keep) >= max_features:
            break
    return keep

@dataclass
class FitResult:
    candidate: str
    fold_name: str
    valid: bool
    pred_ids: list[str]
    pred_y: list[int]
    pred_p: list[float]
    ext_long: list[dict[str, Any]]
    selection: pd.DataFrame
    dimension: int
    params: dict[str, Any]
    status: str

def select_hyperparams(x: np.ndarray, y: np.ndarray, classifier: str) -> tuple[dict[str, Any], Any]:
    skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    candidates = []
    if classifier == "elasticnet":
        for c in [0.1, 1.0, 10.0]:
            for l1 in [0.0, 0.5, 1.0]:
                candidates.append({"C": c, "l1_ratio": l1})
    else:
        for md in [3, 5, None]:
            for leaf in [3, 5, 10]:
                for mf in ["sqrt", 0.5]:
                    candidates.append({"max_depth": md, "min_samples_leaf": leaf, "max_features": mf, "n_estimators": 500})
    scored = []
    for params in candidates:
        fold_scores = []
        for tr, va in skf.split(x, y):
            model = make_model(classifier, params)
            model.fit(x[tr], y[tr])
            p = model.predict_proba(x[va])[:, 1]
            m = metrics(y[va], p)
            fold_scores.append(m)
        scored.append({
            "params": params,
            "AUROC": np.nanmean([s["AUROC"] for s in fold_scores]),
            "AUPRC": np.nanmean([s["AUPRC"] for s in fold_scores]),
            "Brier": np.nanmean([s["Brier"] for s in fold_scores]),
        })
    if classifier == "elasticnet":
        scored.sort(key=lambda r: (-r["AUROC"], -r["AUPRC"], r["Brier"], r["params"]["C"]))
    else:
        scored.sort(key=lambda r: (-r["AUROC"], -r["AUPRC"], r["Brier"], 999 if r["params"]["max_depth"] is None else r["params"]["max_depth"], r["params"]["min_samples_leaf"], str(r["params"]["max_features"])))
    best = scored[0]["params"]
    model = make_model(classifier, best)
    model.fit(x, y)
    return best, model

def make_model(classifier: str, params: dict[str, Any]):
    if classifier == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            class_weight=None,
            max_iter=10000,
            random_state=SEED,
        )
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=None,
        random_state=SEED,
        n_jobs=-1,
    )

def fit_tabular_candidate(candidate: str, classifier: str, df: pd.DataFrame, ext: pd.DataFrame, features: list[str], train_mask: np.ndarray, test_mask: np.ndarray, domains: list[str], max_features: int, fold_name: str, artifact_dir: Path | None = None) -> FitResult:
    selected, sel = meta_select(df, features, train_mask, domains, max_features)
    sel["candidate"] = candidate
    sel["fold_model"] = fold_name
    if len(selected) < 3:
        return FitResult(candidate, fold_name, False, [], [], [], [], sel, 0, {}, "INSUFFICIENT_STABLE_FEATURES")
    train = df.loc[train_mask]
    test = df.loc[test_mask]
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    xtr = scaler.fit_transform(imp.fit_transform(train[selected]))
    ytr = train["label"].astype(int).to_numpy()
    xte = scaler.transform(imp.transform(test[selected]))
    xext = scaler.transform(imp.transform(ext[selected]))
    record_candidate_inputs(artifact_dir,
                            train[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            test[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            ext[["neutral_external_id", "neutral_patient_id"] + selected], xtr, xte, xext)
    params, model = select_hyperparams(xtr, ytr, classifier)
    record_fitted_candidate(artifact_dir, train[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            test[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            ext[["neutral_external_id", "neutral_patient_id"] + selected],
                            selected, xtr, xte, xext, imp, scaler, model, params)
    pte = model.predict_proba(xte)[:, 1]
    pext = model.predict_proba(xext)[:, 1]
    ext_long = [{"neutral_external_id": r["neutral_external_id"], "neutral_patient_id": r["neutral_patient_id"], "candidate": candidate, "fold_model": fold_name, "prob_COP_fold": float(p)} for (_, r), p in zip(ext.iterrows(), pext)]
    return FitResult(candidate, fold_name, True, test["neutral_imaging_id"].astype(str).tolist(), test["label"].astype(int).tolist(), pte.astype(float).tolist(), ext_long, sel, len(selected), params, "PASS")

def embedding_pc_frame(arr: np.ndarray, meta: pd.DataFrame, train_mask: np.ndarray, apply_mask: np.ndarray | None, ext_arr: np.ndarray | None = None, artifact_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str], dict[str, Any]]:
    xtr_raw = arr[train_mask]
    std = StandardScaler()
    xtr_std = std.fit_transform(xtr_raw)
    pca_full = PCA(random_state=SEED)
    pca_full.fit(xtr_std)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = min(int(np.searchsorted(cum, 0.90) + 1), 20, arr.shape[1], arr.shape[0] - 1)
    pca = PCA(n_components=k, random_state=SEED)
    pca.fit(xtr_std)
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(std, artifact_dir / "embedding_scaler.joblib")
        joblib.dump(pca, artifact_dir / "pca.joblib")
        (artifact_dir / "transform_manifest.json").write_text(
            json.dumps({"training_rows": int(train_mask.sum()), "n_components": k,
                        "files_sha256": {p.name: sha256_file(p) for p in sorted(artifact_dir.iterdir()) if p.is_file()}},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_pc = pca.transform(std.transform(arr))
    pc_cols = [f"PC{i+1:02d}" for i in range(k)]
    frame = meta[["neutral_imaging_id", "label", "outer_fold", "primary_domain"]].copy()
    for i, c in enumerate(pc_cols):
        frame[c] = all_pc[:, i]
    ext_frame = None
    if ext_arr is not None:
        ext_pc = pca.transform(std.transform(ext_arr))
        ext_frame = pd.DataFrame({"neutral_external_id": apply_mask["neutral_external_id"].astype(str).tolist(), "neutral_patient_id": apply_mask["neutral_patient_id"].astype(str).tolist()})
        for i, c in enumerate(pc_cols):
            ext_frame[c] = ext_pc[:, i]
    info = {"n_components": k, "variance_explained": float(np.sum(pca.explained_variance_ratio_))}
    return frame, ext_frame, pc_cols, info

def fit_embedding_candidate(candidate: str, family: str, classifier: str, meta: pd.DataFrame, arr: np.ndarray, ext_meta: pd.DataFrame, ext_arr: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, domains: list[str], max_features: int, fold_name: str, artifact_dir: Path | None = None) -> FitResult:
    pc_df, pc_ext, pc_cols, pca_info = embedding_pc_frame(
        arr, meta, train_mask, ext_meta, ext_arr,
        artifact_dir / "pca" if artifact_dir is not None else None)
    selected, sel = meta_select(pc_df, pc_cols, train_mask, domains, max_features, prefix=f"{family}_")
    sel["candidate"] = candidate
    sel["fold_model"] = fold_name
    sel["pca_n_components"] = pca_info["n_components"]
    sel["pca_variance_explained"] = pca_info["variance_explained"]
    if len(selected) < 3:
        return FitResult(candidate, fold_name, False, [], [], [], [], sel, 0, {}, "INSUFFICIENT_STABLE_PCS")
    train = pc_df.loc[train_mask]
    test = pc_df.loc[test_mask]
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    xtr = scaler.fit_transform(imp.fit_transform(train[selected]))
    ytr = train["label"].astype(int).to_numpy()
    xte = scaler.transform(imp.transform(test[selected]))
    xext = scaler.transform(imp.transform(pc_ext[selected]))
    record_candidate_inputs(artifact_dir / "model" if artifact_dir is not None else None,
                            train[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            test[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            pc_ext[["neutral_external_id", "neutral_patient_id"] + selected], xtr, xte, xext)
    params, model = select_hyperparams(xtr, ytr, classifier)
    record_fitted_candidate(artifact_dir / "model" if artifact_dir is not None else None,
                            train[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            test[["neutral_imaging_id", "label", "outer_fold"] + selected],
                            pc_ext[["neutral_external_id", "neutral_patient_id"] + selected],
                            selected, xtr, xte, xext, imp, scaler, model, params | pca_info)
    pte = model.predict_proba(xte)[:, 1]
    pext = model.predict_proba(xext)[:, 1]
    ext_long = [{"neutral_external_id": r["neutral_external_id"], "neutral_patient_id": r["neutral_patient_id"], "candidate": candidate, "fold_model": fold_name, "prob_COP_fold": float(p)} for (_, r), p in zip(pc_ext.iterrows(), pext)]
    return FitResult(candidate, fold_name, True, test["neutral_imaging_id"].astype(str).tolist(), test["label"].astype(int).tolist(), pte.astype(float).tolist(), ext_long, sel, len(selected), params | pca_info, "PASS")

def combine_tabular(a: pd.DataFrame, b: pd.DataFrame, fa: list[str], fb: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cols = ["neutral_imaging_id", "label", "outer_fold", "primary_domain"]
    for col in ["public_subject_id"] + cols:
        if not a[col].reset_index(drop=True).equals(b[col].reset_index(drop=True)):
            raise ValueError(f"Stable feature families have different patient rows: {col}")
    out = a[cols + fa].merge(b[["neutral_imaging_id"] + fb], on="neutral_imaging_id",
                             how="left", validate="one_to_one", sort=False)
    return out, fa + fb

def validate_stable_prefit(base: pd.DataFrame, tab_i: dict[str, pd.DataFrame],
                           tab_e: dict[str, pd.DataFrame], emb_mi: dict[str, pd.DataFrame],
                           emb_me: dict[str, pd.DataFrame], emb_i: dict[str, np.ndarray],
                           emb_e: dict[str, np.ndarray]) -> dict[str, int]:
    """Reject identity or row-order drift before any estimator is fitted."""
    crosswalk = read_csv(PATHS["internal_crosswalk"])
    first_external = next(iter(tab_e.values()))
    ext_index = first_external[["neutral_external_id", "neutral_patient_id"]]
    ci, ce = stable_patient_contract(base, ext_index, crosswalk)
    internal_cols = ["public_subject_id", "neutral_imaging_id", "label", "outer_fold"]
    external_cols = ["public_subject_id", "neutral_external_id", "neutral_patient_id"]
    for family, frame in tab_i.items():
        require_unique_ids(frame["public_subject_id"], f"{family} internal public IDs")
        for col in internal_cols:
            if not frame[col].reset_index(drop=True).astype(str).equals(ci[col].reset_index(drop=True).astype(str)):
                raise ValueError(f"{family}: internal patient, label, fold or row order mismatch: {col}")
    for family, frame in tab_e.items():
        require_unique_ids(frame["public_subject_id"], f"{family} external public IDs")
        for col in external_cols:
            if not frame[col].reset_index(drop=True).astype(str).equals(ce[col].reset_index(drop=True).astype(str)):
                raise ValueError(f"{family}: external patient or row order mismatch: {col}")
    for family, array in emb_i.items():
        meta = emb_mi[family]
        if len(array) != len(ci) or len(meta) != len(ci):
            raise ValueError(f"{family}: internal embedding count mismatch")
        for col in ("neutral_imaging_id", "label", "outer_fold"):
            if not meta[col].reset_index(drop=True).astype(str).equals(ci[col].reset_index(drop=True).astype(str)):
                raise ValueError(f"{family}: internal embedding manifest mismatch: {col}")
    for family, array in emb_e.items():
        meta = emb_me[family]
        if len(array) != len(ce) or len(meta) != len(ce):
            raise ValueError(f"{family}: external embedding count mismatch")
        for col in ("neutral_external_id", "neutral_patient_id"):
            if not meta[col].reset_index(drop=True).astype(str).equals(ce[col].reset_index(drop=True).astype(str)):
                raise ValueError(f"{family}: external embedding manifest mismatch: {col}")
    for fold in sorted(ci["outer_fold"].unique()):
        train = set(ci.loc[ci["outer_fold"].ne(fold), "public_subject_id"])
        held_out = set(ci.loc[ci["outer_fold"].eq(fold), "public_subject_id"])
        if not train or not held_out or train & held_out or train & set(ce["public_subject_id"]):
            raise ValueError("Stable outer split has overlap or an empty partition")
    return {"internal_patients": len(ci), "external_patients": len(ce),
            "internal_identity_mismatches": 0, "external_identity_mismatches": 0,
            "label_mismatches": 0, "fold_mismatches": 0}
