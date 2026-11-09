# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import json

from pathlib import Path

from typing import Any

import numpy as np

import SimpleITK as sitk

from scipy import ndimage

from cap_cop.imaging.radiomics import lesion_phenotype as exact

from cap_cop.imaging.radiomics.mask_contract import validate_locked_mask_contract, validate_mask_metadata

from cap_cop.imaging.radiomics.per_slice_dicom_sampler import (
    read_axial_series,
    sample_normalized_roi,
)

ROI_MATRIX = 144

ROI_SPACING_MM = 184.0 / ROI_MATRIX

HU_MIN = -1000.0

HU_MAX = 400.0

def read_image_lps(folder: Path) -> sitk.Image:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
    if not series_ids:
        raise RuntimeError(f"No DICOM series in {folder}")
    selected = max(series_ids, key=lambda item: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), item)))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), selected))
    return sitk.DICOMOrient(reader.Execute(), "LPS")

def resample_roi_from_native(image: sitk.Image, center_xyz: tuple[float, float, float]) -> np.ndarray:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([ROI_MATRIX, ROI_MATRIX, ROI_MATRIX])
    resampler.SetOutputSpacing([ROI_SPACING_MM] * 3)
    resampler.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    resampler.SetOutputOrigin([center_xyz[index] - ((ROI_MATRIX - 1) / 2.0) * ROI_SPACING_MM for index in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(HU_MIN)
    array_hu = sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.float32)
    normalized = np.clip((np.clip(array_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0).astype(np.float32)
    return normalized[np.newaxis, ...]

def extract_subject(task: dict[str, Any]) -> dict[str, Any]:
    use_per_slice_sampler = bool(task.get("use_per_slice_sampler", False))
    mask = np.load(task["mask_path"]).astype(np.uint8)
    metadata = json.loads(Path(task["metadata_path"]).read_text(encoding="utf-8"))
    component_labels, component_count = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    validate_mask_metadata(task["public_subject_id"], mask.shape, metadata)
    components = exact.component_table(mask, metadata)
    if task["cohort"] == "internal":
        validate_locked_mask_contract(
            task["public_subject_id"], Path(task["mask_path"]),
            component_count, task.get("locked_roi_centers", []),
            mask_shape=mask.shape, metadata=metadata, components=components,
        )
    image = None if use_per_slice_sampler else read_image_lps(Path(task["dicom_dir"]))
    native_slices = read_axial_series(Path(task["dicom_dir"])) if use_per_slice_sampler else None
    if component_count != len(components):
        raise RuntimeError(f"Component count mismatch for {task['public_subject_id']}")

    locked_centers = {
        int(row["predicted_component_rank"]): (
            float(row["center_x_mm"]),
            float(row["center_y_mm"]),
            float(row["center_z_mm"]),
        )
        for row in task.get("locked_roi_centers", [])
    }
    if task["cohort"] == "internal":
        expected_ranks = set(range(1, min(3, len(components)) + 1))
        if set(locked_centers) != expected_ranks:
            raise RuntimeError(
                f"Locked ROI geometry contract mismatch for {task['public_subject_id']}: "
                f"expected={sorted(expected_ranks)} observed={sorted(locked_centers)}"
            )

    component_features = []
    for component in components[:3]:
        rank = int(component["component_rank_by_volume"])
        center = locked_centers.get(
            rank,
            (
                float(component["bbox_center_x_mm"]),
                float(component["bbox_center_y_mm"]),
                float(component["bbox_center_z_mm"]),
            ),
        )
        if native_slices is not None:
            roi = sample_normalized_roi(native_slices, center, ROI_MATRIX, ROI_SPACING_MM)
        else:
            roi = resample_roi_from_native(image, center)
        lesion_mask = exact.roi_component_mask(component_labels, int(component["label"]), center, metadata)
        if not np.any(lesion_mask):
            raise RuntimeError(f"Empty mapped ROI mask for {task['public_subject_id']}")
        features = exact.extract_component_features(roi, lesion_mask)
        component_features.append(
            {
                "component_volume_ml_from_2mm_mask": float(component["volume_ml"]),
                **features,
            }
        )

    patient: dict[str, Any] = {
        "public_subject_id": task["public_subject_id"],
        "neutral_patient_id": task["segmentation_model_id"],
        "source_neutral_imaging_id": task["lesion_imaging_model_id"],
        "neutral_external_id": task["external_tensor_id"],
        "site": "Internal" if task["cohort"] == "internal" else "External",
        "group": ("I_" if task["cohort"] == "internal" else "E_") + task["diagnosis_label"],
        "label_binary": int(task["label_binary"]),
        "radiomics_missing_indicator": 1 if not component_features else 0,
        "predicted_component_count": int(component_count),
        "radiomics_valid_component_count": len(component_features),
        "radiomics_total_component_volume_ml": float(sum(row["component_volume_ml_from_2mm_mask"] for row in component_features)),
        "radiomics_largest_component_volume_ml": float(component_features[0]["component_volume_ml_from_2mm_mask"]) if component_features else np.nan,
        "radiomics_second_component_volume_ml": float(component_features[1]["component_volume_ml_from_2mm_mask"]) if len(component_features) > 1 else np.nan,
        "radiomics_third_component_volume_ml": float(component_features[2]["component_volume_ml_from_2mm_mask"]) if len(component_features) > 2 else np.nan,
    }
    total_volume = patient["radiomics_total_component_volume_ml"]
    patient["radiomics_largest_component_fraction"] = patient["radiomics_largest_component_volume_ml"] / total_volume if total_volume else np.nan
    if component_features:
        weights = np.asarray([row["component_volume_ml_from_2mm_mask"] for row in component_features], dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights), dtype=float) / len(weights)
        names = sorted(set().union(*(set(row) for row in component_features)) - {"component_volume_ml_from_2mm_mask"})
        for name in names:
            values = np.asarray([float(row.get(name, np.nan)) for row in component_features], dtype=float)
            patient[f"largest_{name}"] = values[0]
            valid = np.isfinite(values)
            patient[f"vwm_{name}"] = float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])) if valid.any() else np.nan
    return patient
