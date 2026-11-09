# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

import pydicom

import SimpleITK as sitk

from pydicom.pixel_data_handlers.util import apply_modality_lut

HU_PADDING = -1000.0

HU_MIN = -1000.0

HU_MAX = 400.0

def read_axial_series(dicom_dir: Path) -> dict[str, Any]:
    """Read a locked multi-substack series using its per-slice DICOM geometry."""
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []
    if not series_ids:
        raise RuntimeError(f"No DICOM series in {dicom_dir}")
    selected_files = max(
        (
            list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id))
            for series_id in series_ids
        ),
        key=len,
    )
    slices: list[dict[str, Any]] = []
    for file_name in selected_files:
        dataset = pydicom.dcmread(file_name, force=True)
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
        image = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
        slices.append(
            {
                "path": file_name,
                "ipp": np.asarray(dataset.ImagePositionPatient, dtype=np.float64),
                "iop": orientation,
                "pixel_spacing": np.asarray(dataset.PixelSpacing, dtype=np.float64),
                "hu": image,
            }
        )
    orientations = np.vstack([item["iop"] for item in slices])
    row_direction = orientations[0, :3]
    column_direction = orientations[0, 3:]
    normal = np.cross(row_direction, column_direction)
    normal = normal / np.linalg.norm(normal)
    for item in slices:
        item["normal_position"] = float(np.dot(item["ipp"], normal))
    slices.sort(key=lambda item: item["normal_position"])
    return {
        "slices": slices,
        "row_direction": row_direction,
        "column_direction": column_direction,
        "normal": normal,
        "normal_positions": np.asarray([item["normal_position"] for item in slices], dtype=np.float64),
    }

def _bilinear_sample(
    slice_info: dict[str, Any],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_value: float,
    row_direction: np.ndarray,
    column_direction: np.ndarray,
) -> np.ndarray:
    ipp = slice_info["ipp"]
    row_spacing = float(slice_info["pixel_spacing"][0])
    column_spacing = float(slice_info["pixel_spacing"][1])
    dx = x_grid - ipp[0]
    dy = y_grid - ipp[1]
    dz = z_value - ipp[2]
    column_coordinate = (
        dx * row_direction[0] + dy * row_direction[1] + dz * row_direction[2]
    ) / column_spacing
    row_coordinate = (
        dx * column_direction[0] + dy * column_direction[1] + dz * column_direction[2]
    ) / row_spacing
    rows, columns = slice_info["hu"].shape
    valid = (
        (row_coordinate >= 0)
        & (row_coordinate <= rows - 1)
        & (column_coordinate >= 0)
        & (column_coordinate <= columns - 1)
    )
    row0 = np.floor(np.clip(row_coordinate, 0, rows - 2)).astype(np.int32)
    column0 = np.floor(np.clip(column_coordinate, 0, columns - 2)).astype(np.int32)
    row1 = row0 + 1
    column1 = column0 + 1
    row_weight = np.clip(row_coordinate - row0, 0.0, 1.0).astype(np.float32)
    column_weight = np.clip(column_coordinate - column0, 0.0, 1.0).astype(np.float32)
    array = slice_info["hu"]
    value00 = array[row0, column0]
    value01 = array[row0, column1]
    value10 = array[row1, column0]
    value11 = array[row1, column1]
    sampled = (1 - row_weight) * (
        (1 - column_weight) * value00 + column_weight * value01
    ) + row_weight * ((1 - column_weight) * value10 + column_weight * value11)
    return np.where(valid, sampled, HU_PADDING).astype(np.float32)

def sample_normalized_roi(
    native: dict[str, Any],
    center_xyz: tuple[float, float, float],
    matrix: int,
    spacing_mm: float,
) -> np.ndarray:
    slices = native["slices"]
    positions = native["normal_positions"]
    row_direction = native["row_direction"]
    column_direction = native["column_direction"]
    normal = native["normal"]
    if not (
        abs(float(normal[0])) < 1e-6
        and abs(float(normal[1])) < 1e-6
        and abs(abs(float(normal[2])) - 1.0) < 1e-6
    ):
        raise RuntimeError("Per-slice locked sampler requires an audited axial series")
    x_values = center_xyz[0] - ((matrix - 1) / 2.0) * spacing_mm + np.arange(matrix, dtype=np.float32) * spacing_mm
    y_values = center_xyz[1] - ((matrix - 1) / 2.0) * spacing_mm + np.arange(matrix, dtype=np.float32) * spacing_mm
    z_values = center_xyz[2] - ((matrix - 1) / 2.0) * spacing_mm + np.arange(matrix, dtype=np.float32) * spacing_mm
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    hu = np.empty((matrix, matrix, matrix), dtype=np.float32)
    for z_index, z_value in enumerate(z_values):
        normal_position = float(z_value * normal[2])
        if normal_position < positions[0] or normal_position > positions[-1]:
            hu[z_index] = HU_PADDING
            continue
        high = int(np.searchsorted(positions, normal_position, side="right"))
        if high <= 0:
            low = high = 0
        elif high >= len(positions):
            low = high = len(positions) - 1
        else:
            low = high - 1
        if low == high or abs(float(positions[high] - positions[low])) < 1e-8:
            hu[z_index] = _bilinear_sample(
                slices[low], x_grid, y_grid, float(z_value), row_direction, column_direction
            )
        else:
            weight = np.float32(
                (normal_position - positions[low]) / (positions[high] - positions[low])
            )
            lower = _bilinear_sample(
                slices[low], x_grid, y_grid, float(z_value), row_direction, column_direction
            )
            upper = _bilinear_sample(
                slices[high], x_grid, y_grid, float(z_value), row_direction, column_direction
            )
            valid = (lower != HU_PADDING) & (upper != HU_PADDING)
            hu[z_index] = np.where(valid, (1 - weight) * lower + weight * upper, HU_PADDING)
    normalized = np.clip(hu, HU_MIN, HU_MAX)
    normalized = (normalized - HU_MIN) / (HU_MAX - HU_MIN)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)[np.newaxis]
