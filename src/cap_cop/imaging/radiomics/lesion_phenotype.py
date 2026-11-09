# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import math

from collections import Counter, defaultdict

from typing import Any

import numpy as np

from scipy import ndimage

TENSOR_SHAPE = (1, 144, 144, 144)

ROI_N = 144

ROI_SPACING_MM = 184.0 / 144.0

FIXED_QUANTIZATION_RANGE = (0.0, 1.0)

FIXED_GRAY_LEVELS = 16

def component_table(mask: np.ndarray, meta: dict[str, Any]) -> list[dict[str, Any]]:
    labels, n = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        return []
    origin = np.array(meta["working_origin_xyz"], dtype=float)
    spacing = np.array(meta["working_spacing_xyz"], dtype=float)
    rows: list[dict[str, Any]] = []
    slices = ndimage.find_objects(labels)
    counts = np.bincount(labels.ravel())
    for lab in range(1, n + 1):
        slc = slices[lab - 1]
        if slc is None:
            continue
        z0, z1 = slc[0].start, slc[0].stop - 1
        y0, y1 = slc[1].start, slc[1].stop - 1
        x0, x1 = slc[2].start, slc[2].stop - 1
        center_idx_xyz = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0])
        center_phys = origin + center_idx_xyz * spacing
        voxels = int(counts[lab])
        rows.append(
            {
                "label": lab,
                "voxels": voxels,
                "volume_ml": voxels * float(np.prod(spacing)) / 1000.0,
                "bbox_center_x_mm": float(center_phys[0]),
                "bbox_center_y_mm": float(center_phys[1]),
                "bbox_center_z_mm": float(center_phys[2]),
                "bbox_extent_x_mm": float((x1 - x0 + 1) * spacing[0]),
                "bbox_extent_y_mm": float((y1 - y0 + 1) * spacing[1]),
                "bbox_extent_z_mm": float((z1 - z0 + 1) * spacing[2]),
            }
        )
    # Match the V3.3 bridge/ROI geometry generator exactly: np.argsort(sizes)[::-1].
    rows_by_label = {int(row["label"]): row for row in rows}
    sorted_labels = (np.argsort(counts[1:])[::-1] + 1).tolist()
    rows = [rows_by_label[int(label)] for label in sorted_labels if int(label) in rows_by_label]
    for idx, row in enumerate(rows, start=1):
        row["component_rank_by_volume"] = idx
    return rows

def roi_component_mask(
    comp_labels: np.ndarray,
    comp_label: int,
    center_xyz: tuple[float, float, float],
    meta: dict[str, Any],
) -> np.ndarray:
    origin = np.array(meta["working_origin_xyz"], dtype=float)
    spacing = np.array(meta["working_spacing_xyz"], dtype=float)
    cx, cy, cz = center_xyz
    offset = (np.arange(ROI_N, dtype=np.float64) - (ROI_N - 1) / 2.0) * ROI_SPACING_MM
    xs = cx + offset
    ys = cy + offset
    zs = cz + offset
    xi = np.rint((xs - origin[0]) / spacing[0]).astype(np.int64)
    yi = np.rint((ys - origin[1]) / spacing[1]).astype(np.int64)
    zi = np.rint((zs - origin[2]) / spacing[2]).astype(np.int64)
    valid_x = (xi >= 0) & (xi < comp_labels.shape[2])
    valid_y = (yi >= 0) & (yi < comp_labels.shape[1])
    valid_z = (zi >= 0) & (zi < comp_labels.shape[0])
    safe_x = np.clip(xi, 0, comp_labels.shape[2] - 1)
    safe_y = np.clip(yi, 0, comp_labels.shape[1] - 1)
    safe_z = np.clip(zi, 0, comp_labels.shape[0] - 1)
    sampled = comp_labels[safe_z[:, None, None], safe_y[None, :, None], safe_x[None, None, :]]
    valid = valid_z[:, None, None] & valid_y[None, :, None] & valid_x[None, None, :]
    return (sampled == comp_label) & valid

def quantize(values: np.ndarray) -> np.ndarray:
    lo, hi = FIXED_QUANTIZATION_RANGE
    clipped = np.clip(values, lo, hi)
    q = np.floor((clipped - lo) / max(hi - lo, 1e-12) * FIXED_GRAY_LEVELS).astype(np.int16) + 1
    return np.clip(q, 1, FIXED_GRAY_LEVELS)

def first_order_features(vals: np.ndarray) -> dict[str, float]:
    vals = vals.astype(np.float64, copy=False)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    centered = vals - mean
    p = np.percentile(vals, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(vals, bins=FIXED_GRAY_LEVELS, range=FIXED_QUANTIZATION_RANGE)
    prob = hist.astype(np.float64) / max(hist.sum(), 1)
    prob = prob[prob > 0]
    return {
        "first_mean": mean,
        "first_std": std,
        "first_min": float(np.min(vals)),
        "first_max": float(np.max(vals)),
        "first_median": float(p[2]),
        "first_p10": float(p[0]),
        "first_p25": float(p[1]),
        "first_p75": float(p[3]),
        "first_p90": float(p[4]),
        "first_iqr": float(p[3] - p[1]),
        "first_mad": float(np.mean(np.abs(centered))),
        "first_energy": float(np.sum(vals * vals)),
        "first_entropy": float(-np.sum(prob * np.log2(prob))),
        "first_skew": float(np.mean(centered**3) / (std**3 + 1e-12)),
        "first_kurtosis": float(np.mean(centered**4) / (std**4 + 1e-12)),
    }

def shape_features(mask: np.ndarray, spacing_mm: float) -> dict[str, float]:
    coords = np.argwhere(mask)
    count = int(coords.shape[0])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ext = (maxs - mins + 1).astype(np.float64) * spacing_mm
    volume_ml = count * (spacing_mm**3) / 1000.0
    bbox_volume_ml = float(np.prod(ext) / 1000.0)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface_voxels = int(np.count_nonzero(mask & ~eroded))
    centroid = coords.mean(axis=0)
    patch_center = np.array([(ROI_N - 1) / 2.0] * 3)
    centroid_offset = (centroid - patch_center) * spacing_mm
    return {
        "shape_voxel_count": float(count),
        "shape_volume_ml": float(volume_ml),
        "shape_bbox_extent_x_mm": float(ext[2]),
        "shape_bbox_extent_y_mm": float(ext[1]),
        "shape_bbox_extent_z_mm": float(ext[0]),
        "shape_bbox_volume_ml": bbox_volume_ml,
        "shape_extent_fraction": float(volume_ml / (bbox_volume_ml + 1e-12)),
        "shape_surface_voxel_count": float(surface_voxels),
        "shape_surface_to_volume": float(surface_voxels / max(count, 1)),
        "shape_centroid_offset_x_mm": float(centroid_offset[2]),
        "shape_centroid_offset_y_mm": float(centroid_offset[1]),
        "shape_centroid_offset_z_mm": float(centroid_offset[0]),
    }

def crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    mins = np.maximum(coords.min(axis=0) - 1, 0)
    maxs = np.minimum(coords.max(axis=0) + 2, np.array(mask.shape))
    slc = tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))
    return arr[slc], mask[slc]

def glcm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mat = np.zeros((FIXED_GRAY_LEVELS, FIXED_GRAY_LEVELS), dtype=np.float64)
    for axis in range(3):
        sl1 = [slice(None)] * 3
        sl2 = [slice(None)] * 3
        sl1[axis] = slice(0, -1)
        sl2[axis] = slice(1, None)
        valid = mask[tuple(sl1)] & mask[tuple(sl2)]
        a = q[tuple(sl1)][valid] - 1
        b = q[tuple(sl2)][valid] - 1
        if a.size:
            np.add.at(mat, (a, b), 1)
            np.add.at(mat, (b, a), 1)
    if mat.sum() == 0:
        return {f"glcm_{k}": 0.0 for k in ["contrast", "dissimilarity", "homogeneity", "asm", "energy", "entropy", "correlation"]}
    p = mat / mat.sum()
    i, j = np.indices(p.shape)
    mu_i = float((p * i).sum())
    mu_j = float((p * j).sum())
    sd_i = math.sqrt(float((p * (i - mu_i) ** 2).sum()))
    sd_j = math.sqrt(float((p * (j - mu_j) ** 2).sum()))
    nz = p[p > 0]
    return {
        "glcm_contrast": float((p * (i - j) ** 2).sum()),
        "glcm_dissimilarity": float((p * np.abs(i - j)).sum()),
        "glcm_homogeneity": float((p / (1.0 + (i - j) ** 2)).sum()),
        "glcm_asm": float((p * p).sum()),
        "glcm_energy": float(math.sqrt((p * p).sum())),
        "glcm_entropy": float(-np.sum(nz * np.log2(nz))),
        "glcm_correlation": float(((p * (i - mu_i) * (j - mu_j)).sum()) / (sd_i * sd_j + 1e-12)),
    }

def glrlm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    run_lengths: list[int] = []
    run_grays: list[int] = []
    for axis in range(3):
        moved_q = np.moveaxis(q, axis, -1)
        moved_m = np.moveaxis(mask, axis, -1)
        for idx in np.ndindex(moved_m.shape[:-1]):
            mline = moved_m[idx]
            if not mline.any():
                continue
            qline = moved_q[idx]
            start = None
            last_g = None
            for pos, in_mask in enumerate(mline):
                g = int(qline[pos]) if in_mask else None
                if in_mask and start is None:
                    start = pos
                    last_g = g
                elif in_mask and g != last_g:
                    run_lengths.append(pos - start)
                    run_grays.append(last_g if last_g is not None else 0)
                    start = pos
                    last_g = g
                elif (not in_mask) and start is not None:
                    run_lengths.append(pos - start)
                    run_grays.append(last_g if last_g is not None else 0)
                    start = None
                    last_g = None
            if start is not None:
                run_lengths.append(len(mline) - start)
                run_grays.append(last_g if last_g is not None else 0)
    if not run_lengths:
        return {k: 0.0 for k in ["glrlm_run_count", "glrlm_sre", "glrlm_lre", "glrlm_gln", "glrlm_rln", "glrlm_run_percentage"]}
    lengths = np.asarray(run_lengths, dtype=np.float64)
    grays = np.asarray(run_grays, dtype=np.int16)
    n_runs = float(len(lengths))
    gl_counts = np.array([np.sum(grays == g) for g in range(1, FIXED_GRAY_LEVELS + 1)], dtype=np.float64)
    rl_counts = np.array(list(Counter(lengths.astype(int)).values()), dtype=np.float64)
    return {
        "glrlm_run_count": n_runs,
        "glrlm_sre": float(np.mean(1.0 / (lengths**2))),
        "glrlm_lre": float(np.mean(lengths**2)),
        "glrlm_gln": float(np.sum(gl_counts**2) / n_runs),
        "glrlm_rln": float(np.sum(rl_counts**2) / n_runs),
        "glrlm_run_percentage": float(n_runs / max(np.count_nonzero(mask), 1)),
    }

def glszm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sizes: list[int] = []
    grays: list[int] = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for g in range(1, FIXED_GRAY_LEVELS + 1):
        lab, n = ndimage.label(mask & (q == g), structure=structure)
        if n == 0:
            continue
        counts = np.bincount(lab.ravel())[1:]
        sizes.extend([int(v) for v in counts if v > 0])
        grays.extend([g] * int(np.count_nonzero(counts > 0)))
    if not sizes:
        return {k: 0.0 for k in ["glszm_zone_count", "glszm_sze", "glszm_lze", "glszm_gln", "glszm_zsn", "glszm_zone_percentage"]}
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    grays_arr = np.asarray(grays, dtype=np.int16)
    n_zones = float(len(sizes_arr))
    gl_counts = np.array([np.sum(grays_arr == g) for g in range(1, FIXED_GRAY_LEVELS + 1)], dtype=np.float64)
    size_counts = np.array(list(Counter(sizes_arr.astype(int)).values()), dtype=np.float64)
    return {
        "glszm_zone_count": n_zones,
        "glszm_sze": float(np.mean(1.0 / (sizes_arr**2))),
        "glszm_lze": float(np.mean(sizes_arr**2)),
        "glszm_gln": float(np.sum(gl_counts**2) / n_zones),
        "glszm_zsn": float(np.sum(size_counts**2) / n_zones),
        "glszm_zone_percentage": float(n_zones / max(np.count_nonzero(mask), 1)),
    }

def gldm_ngtdm_features(q: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    kernel = np.ones((3, 3, 3), dtype=np.float64)
    kernel[1, 1, 1] = 0.0
    mask_f = mask.astype(np.float64)
    neighbor_count = ndimage.convolve(mask_f, kernel, mode="constant", cval=0.0)
    q_f = q.astype(np.float64)
    neighbor_sum = ndimage.convolve(q_f * mask_f, kernel, mode="constant", cval=0.0)
    deps = neighbor_count[mask].astype(np.float64)
    if deps.size == 0:
        return {k: 0.0 for k in ["gldm_dependence_mean", "gldm_dependence_std", "gldm_sde", "gldm_lde", "gldm_dn", "ngtdm_coarseness", "ngtdm_contrast", "ngtdm_busyness"]}
    valid_ng = mask & (neighbor_count > 0)
    diffs = np.abs(q_f[valid_ng] - neighbor_sum[valid_ng] / np.maximum(neighbor_count[valid_ng], 1))
    dep_counts = np.array(list(Counter(deps.astype(int)).values()), dtype=np.float64)
    q_vals = q_f[mask]
    sum_diff = float(np.sum(diffs))
    return {
        "gldm_dependence_mean": float(np.mean(deps)),
        "gldm_dependence_std": float(np.std(deps)),
        "gldm_sde": float(np.mean(1.0 / ((deps + 1.0) ** 2))),
        "gldm_lde": float(np.mean((deps + 1.0) ** 2)),
        "gldm_dn": float(np.sum(dep_counts**2) / max(len(deps), 1)),
        "ngtdm_coarseness": float(1.0 / (sum_diff + 1e-12)),
        "ngtdm_contrast": float(np.var(q_vals) * sum_diff / max(len(q_vals), 1)),
        "ngtdm_busyness": float(sum_diff / (np.sum(np.abs(q_vals - np.mean(q_vals))) + 1e-12)),
    }

def extract_component_features(roi: np.ndarray, lesion_mask: np.ndarray) -> dict[str, float]:
    if tuple(roi.shape) != TENSOR_SHAPE:
        raise RuntimeError(f"Bad ROI tensor shape: {roi.shape}")
    arr = roi[0].astype(np.float32, copy=False)
    if lesion_mask.shape != arr.shape:
        raise RuntimeError(f"Mask/ROI shape mismatch: {lesion_mask.shape} vs {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite values in ROI tensor.")
    if lesion_mask.dtype != bool:
        lesion_mask = lesion_mask.astype(bool)
    if np.count_nonzero(lesion_mask) <= 0:
        raise RuntimeError("Mapped lesion mask has zero voxels.")
    cropped_arr, cropped_mask = crop_to_mask(arr, lesion_mask)
    vals = cropped_arr[cropped_mask]
    q = np.zeros(cropped_arr.shape, dtype=np.int16)
    q[cropped_mask] = quantize(vals)
    feats: dict[str, float] = {}
    feats.update(shape_features(cropped_mask, ROI_SPACING_MM))
    feats.update(first_order_features(vals))
    feats.update(glcm_features(q, cropped_mask))
    feats.update(glrlm_features(q, cropped_mask))
    feats.update(glszm_features(q, cropped_mask))
    feats.update(gldm_ngtdm_features(q, cropped_mask))
    return feats
