# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import hashlib

from pathlib import Path

from typing import Any

import numpy as np

import SimpleITK as sitk

from scipy import ndimage

HU_WINDOW = (-1000.0, 400.0)

TARGET_SHAPE_XYZ = (96, 96, 96)

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def series_files(folder: Path) -> tuple[list[str], str]:
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
    if not ids or len(ids) != 1:
        raise RuntimeError(f"Expected one DICOM series, found {0 if not ids else len(ids)}")
    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), ids[0]))
    if not files:
        raise RuntimeError("No DICOM files in series")
    return files, str(ids[0])

def read_image_lps(folder: Path) -> tuple[sitk.Image, dict[str, Any]]:
    files, series_id = series_files(folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()
    original_direction = tuple(float(x) for x in img.GetDirection())
    canonical = sitk.DICOMOrient(img, "LPS")
    canonical_direction = tuple(float(x) for x in canonical.GetDirection())
    direction_pass = bool(np.allclose(np.asarray(canonical_direction).reshape(3, 3), np.eye(3), atol=1e-3))
    meta = {
        "series_id_hash": sha256_text(series_id),
        "dicom_file_count": len(files),
        "original_size_xyz": list(img.GetSize()),
        "canonical_size_xyz": list(canonical.GetSize()),
        "original_spacing_xyz": list(map(float, img.GetSpacing())),
        "canonical_spacing_xyz": list(map(float, canonical.GetSpacing())),
        "original_origin_lps": list(map(float, img.GetOrigin())),
        "canonical_origin_lps": list(map(float, canonical.GetOrigin())),
        "original_direction": list(original_direction),
        "canonical_direction": list(canonical_direction),
        "canonical_transform_status": "PASS" if direction_pass else "FAIL",
    }
    return canonical, meta

def body_proxy_2d(slice_hu: np.ndarray) -> np.ndarray:
    fg = np.asarray(slice_hu > -600, dtype=bool)
    fg = ndimage.binary_fill_holes(fg)
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3), dtype=bool), iterations=1)
    lab, n = ndimage.label(fg)
    if n == 0:
        return np.zeros_like(fg, dtype=bool)
    h, w = fg.shape
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    cx0, cx1 = int(w * 0.35), int(w * 0.65)
    center_labels = lab[cy0:cy1, cx0:cx1]
    labels = [int(x) for x in np.unique(center_labels) if x > 0]
    if not labels:
        counts = np.bincount(lab.ravel())
        labels = [int(np.argmax(counts[1:]) + 1)] if len(counts) > 1 else []
    if not labels:
        return np.zeros_like(fg, dtype=bool)
    counts = {x: int((lab == x).sum()) for x in labels}
    chosen = max(counts, key=counts.get)
    return lab == chosen

def compute_body_proxy(arr_hu_zyx: np.ndarray) -> np.ndarray:
    masks = [body_proxy_2d(arr_hu_zyx[z]) for z in range(arr_hu_zyx.shape[0])]
    return np.stack(masks, axis=0)

def physical_bbox_from_mask(img: sitk.Image, mask_zyx: np.ndarray) -> dict[str, Any]:
    coords = np.argwhere(mask_zyx)
    if coords.size == 0:
        return {"body_proxy_status": "FAIL_EMPTY"}
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)
    pts = [
        img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0))),
        img.TransformIndexToPhysicalPoint((int(x1), int(y1), int(z1))),
    ]
    mn = np.minimum(pts[0], pts[1])
    mx = np.maximum(pts[0], pts[1])
    spacing = np.asarray(img.GetSpacing(), dtype=float)
    ext = (np.asarray([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1], dtype=float) * spacing).tolist()
    cen = ((mn + mx) / 2.0).tolist()
    size = np.asarray(img.GetSize(), dtype=float)
    scan_ext = (size * spacing).tolist()
    scan_center = (np.asarray(img.GetOrigin(), dtype=float) + (size - 1) * spacing / 2.0).tolist()
    return {
        "body_proxy_status": "PASS",
        "bbox_index_x0": int(x0),
        "bbox_index_x1": int(x1),
        "bbox_index_y0": int(y0),
        "bbox_index_y1": int(y1),
        "bbox_index_z0": int(z0),
        "bbox_index_z1": int(z1),
        "body_extent_x_mm": float(ext[0]),
        "body_extent_y_mm": float(ext[1]),
        "body_extent_z_mm": float(ext[2]),
        "body_center_x_mm": float(cen[0]),
        "body_center_y_mm": float(cen[1]),
        "body_center_z_mm": float(cen[2]),
        "original_scan_extent_x_mm": float(scan_ext[0]),
        "original_scan_extent_y_mm": float(scan_ext[1]),
        "original_scan_extent_z_mm": float(scan_ext[2]),
        "scan_center_x_mm": float(scan_center[0]),
        "scan_center_y_mm": float(scan_center[1]),
        "scan_center_z_mm": float(scan_center[2]),
        "center_offset_x_mm": float(cen[0] - scan_center[0]),
        "center_offset_y_mm": float(cen[1] - scan_center[1]),
        "center_offset_z_mm": float(cen[2] - scan_center[2]),
        "body_border_touch_x": bool(x0 == 0 or x1 == mask_zyx.shape[2] - 1),
        "body_border_touch_y": bool(y0 == 0 or y1 == mask_zyx.shape[1] - 1),
        "body_border_touch_z": bool(z0 == 0 or z1 == mask_zyx.shape[0] - 1),
        "body_proxy_voxels": int(mask_zyx.sum()),
        "body_proxy_fraction_of_scan": float(mask_zyx.mean()),
        "body_volume_proxy_ml": float(mask_zyx.sum() * np.prod(spacing) / 1000.0),
    }

def output_origin(center_xyz: tuple[float, float, float], fov_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    spacing = np.asarray(fov_xyz, dtype=float) / 96.0
    return tuple((np.asarray(center_xyz, dtype=float) - np.asarray(fov_xyz, dtype=float) / 2.0 + spacing / 2.0).tolist())

def resample_tensor(img_lps: sitk.Image, center_xyz: tuple[float, float, float], fov_xyz: tuple[float, float, float], use_antialias: bool = True) -> tuple[np.ndarray, dict[str, float]]:
    spacing_native = np.asarray(img_lps.GetSpacing(), dtype=float)
    target_spacing = np.asarray(fov_xyz, dtype=float) / 96.0
    factor = target_spacing / spacing_native
    sigma_xyz = np.where(factor > 1.0, (factor - 1.0) / 2.0, 0.0)
    img_for_resample = img_lps
    if use_antialias and np.any(sigma_xyz > 0):
        arr = sitk.GetArrayFromImage(img_lps).astype(np.float32)
        sigma_zyx = (float(sigma_xyz[2]), float(sigma_xyz[1]), float(sigma_xyz[0]))
        arr = ndimage.gaussian_filter(arr, sigma=sigma_zyx, mode="nearest")
        img_for_resample = sitk.GetImageFromArray(arr)
        img_for_resample.CopyInformation(img_lps)
    res = sitk.ResampleImageFilter()
    res.SetSize(TARGET_SHAPE_XYZ)
    res.SetOutputSpacing(tuple(target_spacing.tolist()))
    res.SetOutputOrigin(output_origin(center_xyz, fov_xyz))
    res.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    res.SetInterpolator(sitk.sitkLinear)
    res.SetDefaultPixelValue(-1000.0)
    out = res.Execute(img_for_resample)
    hu = sitk.GetArrayFromImage(out).astype(np.float32)
    scaled = np.clip((np.clip(hu, HU_WINDOW[0], HU_WINDOW[1]) - HU_WINDOW[0]) / (HU_WINDOW[1] - HU_WINDOW[0]), 0.0, 1.0).astype(np.float32)
    params = {
        "native_spacing_x_mm": float(spacing_native[0]),
        "native_spacing_y_mm": float(spacing_native[1]),
        "native_spacing_z_mm": float(spacing_native[2]),
        "target_spacing_x_mm": float(target_spacing[0]),
        "target_spacing_y_mm": float(target_spacing[1]),
        "target_spacing_z_mm": float(target_spacing[2]),
        "downsampling_factor_x": float(factor[0]),
        "downsampling_factor_y": float(factor[1]),
        "downsampling_factor_z": float(factor[2]),
        "sigma_x_native_voxels": float(sigma_xyz[0] if use_antialias else 0.0),
        "sigma_y_native_voxels": float(sigma_xyz[1] if use_antialias else 0.0),
        "sigma_z_native_voxels": float(sigma_xyz[2] if use_antialias else 0.0),
    }
    return scaled[None, ...], params
