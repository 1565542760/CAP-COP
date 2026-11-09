# Final-method dependency subset; retained scientific definitions are source-verbatim.
"""Validate that locked lesion ROI geometry belongs to the supplied mask."""

from __future__ import annotations

import hashlib

import math

from pathlib import Path

def mask_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

def validate_locked_mask_contract(
    public_subject_id: str,
    mask_path: Path,
    component_count: int,
    geometry_rows: list[dict[str, str]],
    *,
    mask_shape: tuple[int, ...] | None = None,
    metadata: dict | None = None,
    components: list[dict] | None = None,
) -> None:
    if not geometry_rows:
        raise RuntimeError(f"Missing locked ROI geometry for {public_subject_id}")
    expected_hashes = {str(row["prediction_mask_sha256"]).upper() for row in geometry_rows}
    expected_counts = {int(row["predicted_component_count"]) for row in geometry_rows}
    if len(expected_hashes) != 1 or len(expected_counts) != 1:
        raise RuntimeError(f"Inconsistent locked ROI geometry for {public_subject_id}")
    if mask_sha256(mask_path) not in expected_hashes:
        raise RuntimeError(
            f"Mask version differs from locked ROI geometry for {public_subject_id}; "
            "do not reuse locked ROI centres with this mask"
        )
    if component_count not in expected_counts:
        raise RuntimeError(
            f"Component count differs from locked ROI geometry for {public_subject_id}: "
            f"mask={component_count}, geometry={next(iter(expected_counts))}"
        )
    if metadata is None or mask_shape is None or components is None:
        return
    validate_mask_metadata(public_subject_id, mask_shape, metadata)
    if len(components) != component_count:
        raise RuntimeError("Computed components disagree with mask component count")
    by_rank = {int(item["component_rank_by_volume"]): item for item in components}
    ranks = [int(row["predicted_component_rank"]) for row in geometry_rows]
    if len(ranks) != len(set(ranks)) or set(ranks) != set(range(1, min(3, component_count) + 1)):
        raise RuntimeError("Locked ROI ranks disagree with current mask components")
    for row in geometry_rows:
        component = by_rank[int(row["predicted_component_rank"])]
        origin = [float(value) for value in metadata["working_origin_xyz"]]
        direction = [float(value) for value in metadata["working_direction"]]
        axis_center = [float(component[f"bbox_center_{axis}_mm"]) for axis in "xyz"]
        displacement = [axis_center[index] - origin[index] for index in range(3)]
        physical_center = [
            origin[index] + sum(direction[3 * index + axis] * displacement[axis] for axis in range(3))
            for index in range(3)
        ]
        for index, axis in enumerate("xyz"):
            locked = float(row[f"center_{axis}_mm"])
            current = physical_center[index]
            if not math.isfinite(locked) or not math.isclose(locked, current, abs_tol=1e-5):
                raise RuntimeError(f"Locked ROI centre disagrees with current mask geometry on {axis}")

def validate_mask_metadata(public_subject_id: str, mask_shape: tuple[int, ...], metadata: dict) -> None:
    if metadata.get("public_subject_id") != public_subject_id:
        raise RuntimeError("Mask metadata patient ID disagrees with requested patient")
    if tuple(metadata.get("shape_zyx", ())) != tuple(mask_shape) or len(mask_shape) != 3:
        raise RuntimeError("Mask shape disagrees with segmentation metadata")
    for key, length in (("working_spacing_xyz", 3), ("working_origin_xyz", 3), ("working_direction", 9)):
        values = metadata.get(key)
        if not isinstance(values, list) or len(values) != length or not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError(f"Mask metadata has invalid {key}")
    if any(float(value) <= 0 for value in metadata["working_spacing_xyz"]):
        raise RuntimeError("Mask metadata has invalid spacing")
    # The locked component-to-ROI transform assumes axis-aligned LPS coordinates.
    expected = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    if any(not math.isclose(float(a), b, abs_tol=1e-6) for a, b in zip(metadata["working_direction"], expected)):
        raise RuntimeError("Mask direction is incompatible with the locked ROI transform")
    if not math.isclose(float(metadata.get("threshold", float("nan"))), 0.5, abs_tol=1e-8):
        raise RuntimeError("Mask threshold differs from the locked segmentation threshold")
