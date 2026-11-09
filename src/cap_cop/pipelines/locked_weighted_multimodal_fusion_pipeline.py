# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import pandas as pd

WEIGHTS = [round(i / 20, 2) for i in range(21)]

def metric_values(y: Iterable[int], p: Iterable[float]) -> Dict[str, float]:
    y_arr = np.asarray(list(y), dtype=int)
    p_arr = np.clip(np.asarray(list(p), dtype=float), 1e-15, 1 - 1e-15)
    pos = y_arr == 1
    neg = y_arr == 0
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("Both classes required for AUROC")
    order = np.argsort(p_arr)
    ranks = np.empty(len(p_arr), dtype=float)
    sorted_p = p_arr[order]
    start = 0
    while start < len(p_arr):
        end = start + 1
        while end < len(p_arr) and sorted_p[end] == sorted_p[start]:
            end += 1
        avg = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg
        start = end
    auroc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    desc = np.argsort(-p_arr, kind="mergesort")
    y_desc = y_arr[desc]
    tp = np.cumsum(y_desc == 1)
    fp = np.cumsum(y_desc == 0)
    precision = tp / np.maximum(tp + fp, 1)
    auprc = precision[y_desc == 1].sum() / pos.sum()
    brier = np.mean((p_arr - y_arr) ** 2)
    ll = -np.mean(y_arr * np.log(p_arr) + (1 - y_arr) * np.log(1 - p_arr))
    return {"AUROC": float(auroc), "AUPRC": float(auprc), "Brier": float(brier), "LogLoss": float(ll)}

def select_weight_from_df(df: pd.DataFrame) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    rows = []
    y = df["label"].astype(int).to_numpy()
    pc = df["p_clinical"].astype(float).to_numpy()
    pi = df["p_image"].astype(float).to_numpy()
    for w in WEIGHTS:
        p = w * pc + (1 - w) * pi
        m = metric_values(y, p)
        rows.append({
            "weight_clinical": w,
            "weight_image": 1 - w,
            **m,
            "distance_to_0_5": abs(w - 0.5),
        })
    score = pd.DataFrame(rows)
    max_auc = score["AUROC"].max()
    cur = score[score["AUROC"] >= max_auc - 0.002].copy()
    max_auprc = cur["AUPRC"].max()
    cur = cur[cur["AUPRC"] >= max_auprc - 0.005].copy()
    min_brier = cur["Brier"].min()
    cur = cur[cur["Brier"] <= min_brier + 1e-12].copy()
    cur = cur.sort_values(["distance_to_0_5", "weight_clinical"], ascending=[True, False])
    sel = cur.iloc[0].to_dict()
    tie_status = {
        "max_AUROC": float(max_auc),
        "AUROC_near_tie_count": int((score["AUROC"] >= max_auc - 0.002).sum()),
        "post_AUPRC_tie_count": int((score[(score["AUROC"] >= max_auc - 0.002)]["AUPRC"] >= max_auprc - 0.005).sum()),
        "selected_weight": float(sel["weight_clinical"]),
        "selection_score_AUROC": float(sel["AUROC"]),
        "selection_score_AUPRC": float(sel["AUPRC"]),
        "selection_score_Brier": float(sel["Brier"]),
    }
    return float(sel["weight_clinical"]), tie_status, score
