# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

import torch

import torch.nn.functional as F

from cap_cop.vendor.tangerine_3d_mae import models_vit

def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise RuntimeError("No tensor state_dict found in TANGERINE checkpoint")

def load_foundation_encoder(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(checkpoint)
    model = models_vit.vit_large_patch16_yo(num_classes=1, global_pool=True, img_size=256)
    model_state = model.state_dict()
    loadable = {
        key: value
        for key, value in state.items()
        if key in model_state
        and torch.is_tensor(value)
        and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model.load_state_dict(loadable, strict=False)
    model.eval().to(device)
    return model

def prepare_tensor_for_encoder(tensor_path: Path, device: torch.device) -> torch.Tensor:
    arr = np.load(tensor_path).astype(np.float32, copy=False)
    if arr.shape == (1, 96, 96, 96):
        arr = arr[None, ...]
    elif arr.shape == (96, 96, 96):
        arr = arr[None, None, ...]
    elif arr.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected tensor shape {arr.shape} for {tensor_path}")
    x = torch.from_numpy(arr)
    x = F.interpolate(x, size=(256, 256, 256), mode="trilinear", align_corners=False)
    return x.to(device=device, non_blocking=False)

@torch.no_grad()
def extract_foundation_embedding(model: torch.nn.Module, tensor_path: Path, device: torch.device) -> np.ndarray:
    x = prepare_tensor_for_encoder(tensor_path, device)
    if x.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            emb = model.forward_features(x)
    else:
        emb = model.forward_features(x)
    emb_np = emb.detach().float().cpu().numpy()
    if emb_np.shape != (1, 2048):
        raise RuntimeError(f"Unexpected embedding shape {emb_np.shape}; expected (1, 2048)")
    return emb_np.astype(np.float32, copy=False)
