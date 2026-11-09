# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

import torch

import torch.nn as nn

from cap_cop.vendor.models_genesis import unet3d as genesis_unet3d

def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint)
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    return {
        str(k).replace("module.", "").replace("model.", "").replace("encoder.", ""): v
        for k, v in state.items()
        if torch.is_tensor(v)
    }

def extract_genesis_embedding(tensor_path: Path, checkpoint_path: Path, device: torch.device) -> np.ndarray:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(raw)
    model = genesis_unet3d.UNet3D(n_class=1, act="relu")
    model_state = model.state_dict()
    keep = {k: v for k, v in state.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
    model.load_state_dict(keep, strict=False)
    model.to(device)
    model.eval()
    pool = nn.AdaptiveAvgPool3d(1)

    arr = np.load(tensor_path).astype(np.float32, copy=False)
    if arr.ndim == 3:
        arr = arr[None, None]
    elif arr.ndim == 4:
        arr = arr[None]
    elif arr.ndim != 5:
        raise ValueError(f"Unexpected tensor shape for Genesis: {arr.shape}")
    x = torch.from_numpy(arr).to(device)
    with torch.no_grad():
        out64, _ = model.down_tr64(x)
        out128, _ = model.down_tr128(out64)
        out256, _ = model.down_tr256(out128)
        out512, _ = model.down_tr512(out256)
        emb = pool(out512).flatten(1).detach().cpu().numpy()
    if emb.shape[0] != 1:
        raise RuntimeError(f"Unexpected Genesis embedding shape: {emb.shape}")
    return emb.astype(np.float32, copy=False)
