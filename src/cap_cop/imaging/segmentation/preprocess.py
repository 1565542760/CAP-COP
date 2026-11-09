# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from dataclasses import dataclass

import math

from pathlib import Path

import numpy as np

from scipy import ndimage

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None

HU_MIN = -1000.0

HU_MAX = 400.0

WORKING_SPACING_XYZ = (2.0, 2.0, 2.0)

@dataclass(frozen=True)
class PreprocessedVolume:
    array_zyx: np.ndarray
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]
    shape_zyx: tuple[int, int, int]

def normalise_hu(array_hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(array_hu.astype(np.float32), HU_MIN, HU_MAX)
    normalised = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    return np.clip(normalised, 0.0, 1.0).astype(np.float32)

def read_primary_dicom_series(dicom_dir: str | Path):
    if sitk is None:
        raise RuntimeError("SimpleITK is required for DICOM reading.")
    path = Path(dicom_dir)
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or []
    if len(series_ids) != 1:
        raise ValueError(f"Expected one readable DICOM series in {path}; found {len(series_ids)}")
    series_id = series_ids[0]
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_id)
    if not files:
        raise ValueError(f"No DICOM files found for series {series_id} in {path}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = sitk.DICOMOrient(reader.Execute(), "LPS")
    return image, series_id, list(files)

def _resample_to_working_spacing(image):
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    size = np.asarray(image.GetSize(), dtype=float)
    extent = (size - 1.0) * spacing
    new_size = np.asarray(
        [int(math.ceil(value / target)) + 1 for value, target in zip(extent, WORKING_SPACING_XYZ)],
        dtype=int,
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(WORKING_SPACING_XYZ)
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(HU_MIN)
    return resampler.Execute(image)

def resample_hu_image_to_working_spacing(image):
    """Apply the locked smoothing and 2-mm resampling to an LPS HU image."""
    original_spacing = tuple(float(v) for v in image.GetSpacing())
    original_origin = tuple(float(v) for v in image.GetOrigin())
    original_direction = tuple(float(v) for v in image.GetDirection())
    factor_xyz = np.asarray(WORKING_SPACING_XYZ, dtype=float) / np.asarray(
        image.GetSpacing(), dtype=float
    )
    sigma_xyz = np.maximum((factor_xyz - 1.0) / 2.0, 0.0)
    array_hu = sitk.GetArrayFromImage(image).astype(np.float32)
    sigma_zyx = (float(sigma_xyz[2]), float(sigma_xyz[1]), float(sigma_xyz[0]))
    if max(sigma_zyx) > 0:
        array_hu = ndimage.gaussian_filter(array_hu, sigma=sigma_zyx, mode="nearest")
        image = sitk.GetImageFromArray(array_hu)
        image.SetSpacing(original_spacing)
        image.SetOrigin(original_origin)
        image.SetDirection(original_direction)
    return _resample_to_working_spacing(image)

def preprocess_dicom_for_segmentation(dicom_dir: str | Path) -> PreprocessedVolume:
    image, _series_id, _files = read_primary_dicom_series(dicom_dir)
    resampled = resample_hu_image_to_working_spacing(image)
    array = normalise_hu(sitk.GetArrayFromImage(resampled))
    return PreprocessedVolume(
        array_zyx=array,
        spacing_xyz=tuple(float(v) for v in resampled.GetSpacing()),
        origin_xyz=tuple(float(v) for v in resampled.GetOrigin()),
        direction=tuple(float(v) for v in resampled.GetDirection()),
        shape_zyx=tuple(int(v) for v in array.shape),
    )
