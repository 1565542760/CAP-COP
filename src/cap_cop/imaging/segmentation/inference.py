# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from scipy import ndimage

from cap_cop.imaging.segmentation.model import load_locked_checkpoint

from cap_cop.paths import project_root

PATCH_ZYX = (128, 128, 128)

THRESHOLD = 0.5

def gaussian_importance(shape: tuple[int, int, int] = PATCH_ZYX) -> torch.Tensor:
    array = np.zeros(shape, dtype=np.float32)
    center = tuple(size // 2 for size in shape)
    array[center] = 1.0
    array = ndimage.gaussian_filter(array, sigma=[size / 8.0 for size in shape])
    array = array / max(float(array.max()), 1e-8)
    array = np.maximum(array, 1e-3).astype(np.float32)
    return torch.from_numpy(array)

def sliding_starts(dim: int, win: int = 128, step: int = 64) -> list[int]:
    if dim <= win:
        return [0]
    starts = list(range(0, dim - win + 1, step))
    if starts[-1] != dim - win:
        starts.append(dim - win)
    return starts

@torch.no_grad()
def predict_volume_logits(model: torch.nn.Module, ct_zyx: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    z, y, x = ct_zyx.shape
    pad_z = max(0, PATCH_ZYX[0] - z)
    pad_y = max(0, PATCH_ZYX[1] - y)
    pad_x = max(0, PATCH_ZYX[2] - x)
    array = np.pad(ct_zyx, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0).astype(np.float32)
    output = np.zeros(array.shape, dtype=np.float32)
    weight_sum = np.zeros(array.shape, dtype=np.float32)
    importance = gaussian_importance().to(device)
    importance_np = importance.detach().cpu().numpy()
    amp = device.type == "cuda"
    for zz in sliding_starts(array.shape[0]):
        for yy in sliding_starts(array.shape[1]):
            for xx in sliding_starts(array.shape[2]):
                patch = torch.from_numpy(array[zz:zz + 128, yy:yy + 128, xx:xx + 128][None, None]).to(device)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(patch)[0, 0].float()
                logits_np = logits.detach().cpu().numpy()
                output[zz:zz + 128, yy:yy + 128, xx:xx + 128] += logits_np * importance_np
                weight_sum[zz:zz + 128, yy:yy + 128, xx:xx + 128] += importance_np
    output = output / np.maximum(weight_sum, 1e-6)
    return output[:z, :y, :x]

def checkpoint_manifest_path() -> Path:
    return project_root() / "models" / "segmentation" / "manifests" / "locked_checkpoint_manifest.csv"

def locked_checkpoint_dir() -> Path:
    return project_root() / "models" / "segmentation" / "locked_checkpoints"

def load_checkpoint_manifest() -> pd.DataFrame:
    frame = pd.read_csv(checkpoint_manifest_path(), dtype=str)
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    frame["inner_fold"] = frame["inner_fold"].astype(int)
    frame["local_checkpoint_path"] = frame["run_id"].astype(str) + "_best_checkpoint.pt"
    return frame.sort_values(["outer_fold", "inner_fold"]).reset_index(drop=True)

def checkpoint_paths_for_outer(outer_fold: int) -> list[Path]:
    rows = load_checkpoint_manifest()
    selected = rows[rows["outer_fold"].eq(int(outer_fold))]
    if len(selected) != 4:
        raise ValueError(f"Expected four inner checkpoints for outer fold {outer_fold}; found {len(selected)}.")
    return [locked_checkpoint_dir() / row.local_checkpoint_path for row in selected.itertuples(index=False)]

def all_checkpoint_paths() -> list[Path]:
    rows = load_checkpoint_manifest()
    if len(rows) != 20:
        raise ValueError(f"Expected 20 locked checkpoints; found {len(rows)}.")
    return [locked_checkpoint_dir() / row.local_checkpoint_path for row in rows.itertuples(index=False)]

def ensemble_probability(ct_zyx: np.ndarray, checkpoint_paths: list[Path], device: torch.device, max_models: int | None = None) -> np.ndarray:
    selected = checkpoint_paths[:max_models] if max_models else checkpoint_paths
    if not selected:
        raise ValueError("No checkpoints selected for inference.")
    accumulator = np.zeros(ct_zyx.shape, dtype=np.float32)
    for path in selected:
        model = load_locked_checkpoint(path, device=device)
        logits = predict_volume_logits(model, ct_zyx, device)
        accumulator += torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32, copy=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return accumulator / float(len(selected))

def ensemble_probability_from_models(
    ct_zyx: np.ndarray,
    models: list[torch.nn.Module],
    device: torch.device,
) -> np.ndarray:
    """Apply an already-loaded locked ensemble without changing its averaging rule."""
    if not models:
        raise ValueError("No loaded models supplied for inference.")
    accumulator = np.zeros(ct_zyx.shape, dtype=np.float32)
    for model in models:
        logits = predict_volume_logits(model, ct_zyx, device)
        accumulator += torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32, copy=False)
    return accumulator / float(len(models))

def mask_from_probability(probability: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return (probability >= threshold).astype(np.uint8)
