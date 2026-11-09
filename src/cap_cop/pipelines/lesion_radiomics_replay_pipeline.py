# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import hashlib

import math

from pathlib import Path

from typing import Any, TYPE_CHECKING

import numpy as np

import pandas as pd

import SimpleITK as sitk

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from cap_cop.data.model_contracts import align_radiomics_training_inputs, require_binary_classes

SEED = 20260816

THRESHOLD = 0.5

ICC_THRESHOLD = 0.90

PERTURBATIONS = [
    "BASELINE",
    "SPACING_1.0MM",
    "SPACING_2.0MM",
    "SMOOTH_0.5MM",
    "SMOOTH_1.0MM",
    "NOISE_10HU",
    "NOISE_20HU",
    "HU_OFFSET_NEG10",
    "HU_OFFSET_POS10",
    "HU_SCALE_0.95",
    "HU_SCALE_1.05",
    "MASK_EROSION_1MM",
    "MASK_DILATION_1MM",
]

CLASSES = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]

def extractor_for_spacing(spacing: float = 1.5) -> featureextractor.RadiomicsFeatureExtractor:
    # Matrix-only modelling does not require the extraction binary extension.
    from radiomics import featureextractor

    params = {
        "imageType": {"Original": {}},
        "featureClass": {k: [] for k in CLASSES},
        "setting": {
            "binWidth": 25,
            "resampledPixelSpacing": [spacing, spacing, spacing],
            "interpolator": "sitkBSpline",
            "normalize": False,
            "label": 1,
        },
    }
    return featureextractor.RadiomicsFeatureExtractor(params)

def load_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        files = sorted(str(p) for p in dicom_dir.glob("*.dcm"))
    else:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    if not files:
        raise RuntimeError(f"No DICOM files found for {dicom_dir}")
    reader.SetFileNames(files)
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)

def image_from_array_like(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr.astype(np.float32, copy=False))
    out.CopyInformation(ref)
    return out

def mask_from_npy(mask_path: Path, meta: dict[str, Any]) -> sitk.Image:
    arr = np.load(mask_path).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(x) for x in meta["working_spacing_xyz"]))
    img.SetOrigin(tuple(float(x) for x in meta["working_origin_xyz"]))
    img.SetDirection(tuple(float(x) for x in meta["working_direction"]))
    return img

def perturb(image: sitk.Image, mask: sitk.Image, scenario: str, seed: int) -> tuple[sitk.Image, sitk.Image, float]:
    if scenario == "BASELINE":
        return image, mask, 1.5
    if scenario == "SPACING_1.0MM":
        return image, mask, 1.0
    if scenario == "SPACING_2.0MM":
        return image, mask, 2.0
    if scenario == "SMOOTH_0.5MM":
        return sitk.SmoothingRecursiveGaussian(image, 0.5), mask, 1.5
    if scenario == "SMOOTH_1.0MM":
        return sitk.SmoothingRecursiveGaussian(image, 1.0), mask, 1.5
    arr = None
    if scenario in {"NOISE_10HU", "NOISE_20HU", "HU_OFFSET_NEG10", "HU_OFFSET_POS10", "HU_SCALE_0.95", "HU_SCALE_1.05"}:
        arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=True)
    if scenario == "NOISE_10HU":
        rng = np.random.default_rng(seed)
        return image_from_array_like(arr + rng.normal(0.0, 10.0, arr.shape).astype(np.float32), image), mask, 1.5
    if scenario == "NOISE_20HU":
        rng = np.random.default_rng(seed)
        return image_from_array_like(arr + rng.normal(0.0, 20.0, arr.shape).astype(np.float32), image), mask, 1.5
    if scenario == "HU_OFFSET_NEG10":
        return image_from_array_like(arr - 10.0, image), mask, 1.5
    if scenario == "HU_OFFSET_POS10":
        return image_from_array_like(arr + 10.0, image), mask, 1.5
    if scenario == "HU_SCALE_0.95":
        return image_from_array_like(arr * 0.95, image), mask, 1.5
    if scenario == "HU_SCALE_1.05":
        return image_from_array_like(arr * 1.05, image), mask, 1.5
    if scenario in {"MASK_EROSION_1MM", "MASK_DILATION_1MM"}:
        dist = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
        if scenario == "MASK_EROSION_1MM":
            m2 = sitk.BinaryThreshold(dist, lowerThreshold=1.0, upperThreshold=1e9, insideValue=1, outsideValue=0)
        else:
            m2 = sitk.BinaryThreshold(dist, lowerThreshold=-1.0, upperThreshold=1e9, insideValue=1, outsideValue=0)
        return image, sitk.Cast(m2, sitk.sitkUInt8), 1.5
    raise RuntimeError(f"Unknown perturbation {scenario}")

def extract_features(image: sitk.Image, mask: sitk.Image, spacing: float) -> dict[str, float]:
    res = extractor_for_spacing(spacing).execute(image, mask)
    out: dict[str, float] = {}
    for k, v in res.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out

def patient_seed(pid: str, scenario: str) -> int:
    return int(hashlib.sha256(f"{SEED}|{pid}|{scenario}".encode("utf-8")).hexdigest()[:8], 16)

def feature_cols(matrix: pd.DataFrame) -> list[str]:
    meta = {"public_subject_id", "neutral_patient_id", "source_neutral_imaging_id",
            "outer_fold", "perturbation", "status", "error"}
    return [c for c in matrix.columns if c not in meta]

def feature_class(name: str) -> str:
    parts = name.split("_")
    return parts[1] if len(parts) > 2 and parts[0] == "original" else "unknown"

def icc_absolute(df: pd.DataFrame, feature: str, patients: list[str] | None = None) -> tuple[float, float]:
    sub = df[["neutral_patient_id", "perturbation", feature, "status"]].copy()
    if patients is not None:
        sub = sub[sub["neutral_patient_id"].isin(patients)]
    sub = sub[sub["status"] == "PASS"]
    wide = sub.pivot_table(index="neutral_patient_id", columns="perturbation", values=feature, aggfunc="first")
    wide = wide.reindex(columns=PERTURBATIONS)
    fail_rate = float(1.0 - wide.notna().all(axis=1).mean()) if len(wide) else 1.0
    wide = wide.dropna()
    if wide.shape[0] < 3:
        return float("nan"), fail_rate
    vals = wide.to_numpy(dtype=float)
    n, k = vals.shape
    ms_rows = k * np.var(vals.mean(axis=1), ddof=1)
    ms_cols = n * np.var(vals.mean(axis=0), ddof=1)
    residual = vals - vals.mean(axis=1, keepdims=True) - vals.mean(axis=0, keepdims=True) + vals.mean()
    ms_err = float(np.sum(residual ** 2) / max((n - 1) * (k - 1), 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    icc = float((ms_rows - ms_err) / denom) if abs(denom) > 1e-12 else float("nan")
    return icc, fail_rate

def stability_table(matrix: pd.DataFrame, patients: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for feat in feature_cols(matrix):
        icc, fail = icc_absolute(matrix, feat, patients)
        rows.append({
            "feature": feat,
            "feature_class": feature_class(feat),
            "ICC_absolute_agreement": icc,
            "technical_failure_rate": fail,
            "ROBUST_ICC_GE_0_90": bool(math.isfinite(icc) and icc >= ICC_THRESHOLD and fail <= 0.05),
        })
    return pd.DataFrame(rows).sort_values(["ROBUST_ICC_GE_0_90", "ICC_absolute_agreement"], ascending=[False, False])

def reduce_redundancy(X: pd.DataFrame, stab: pd.DataFrame) -> list[str]:
    robust = stab[stab["ROBUST_ICC_GE_0_90"]]["feature"].astype(str).tolist()
    if not robust:
        return []
    keep = []
    drop = set()
    ranked = stab.set_index("feature").loc[robust].sort_values(["ICC_absolute_agreement", "technical_failure_rate"], ascending=[False, True]).index.tolist()
    corr = X[robust].corr(method="spearman").abs()
    for f in ranked:
        if f in drop:
            continue
        keep.append(f)
        high = corr.index[(corr[f] > 0.90) & (corr.index != f)].tolist()
        for h in high:
            if h not in keep:
                drop.add(h)
    return keep

def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= THRESHOLD).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Sensitivity": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "Specificity": float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }

def build_model(kind: str, params: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "LR":
        steps.append(("scaler", StandardScaler()))
        steps.append(("model", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            class_weight=params["class_weight"],
            max_iter=5000,
            random_state=SEED,
        )))
    else:
        steps.append(("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=SEED,
            n_jobs=-1,
        )))
    return Pipeline(steps)

def model_grid() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for C in [0.01, 0.1, 1, 10]:
        for l1_ratio in [0, 0.25, 0.5, 0.75, 1]:
            for cw in [None, "balanced"]:
                out.append(("LR", {"C": C, "l1_ratio": l1_ratio, "class_weight": cw}))
    for md in [3, 5, None]:
        for leaf in [3, 5, 10]:
            for mf in ["sqrt", 0.5]:
                for cw in [None, "balanced"]:
                    out.append(("RF", {"max_depth": md, "min_samples_leaf": leaf, "max_features": mf, "class_weight": cw}))
    return out

def rank_features(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> list[str]:
    scores = []
    yv = y.to_numpy(dtype=int)
    for f in feats:
        x = pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 5 or len(np.unique(x[mask])) < 2:
            score = 0.5
        else:
            try:
                auc = roc_auc_score(yv[mask], x[mask])
                score = max(auc, 1 - auc)
            except Exception:
                score = 0.5
        scores.append((f, score))
    return [f for f, _ in sorted(scores, key=lambda x: (-x[1], x[0]))]

def select_best_candidate(score_df: pd.DataFrame) -> pd.Series:
    agg = score_df.groupby(["kind", "params", "n_features"], as_index=False).agg({
        "AUROC": "mean",
        "AUPRC": "mean",
        "Brier": "mean",
        "feature_set": "first",
    })
    max_auc = agg["AUROC"].max()
    guard = agg[agg["AUROC"] >= max_auc - 0.01].copy()
    guard = guard.sort_values(["AUROC", "AUPRC", "Brier", "n_features", "kind"], ascending=[False, False, True, True, True])
    return guard.iloc[0]
