# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import argparse

import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

from cap_cop.imaging.whole_chest_tensor import locked_p95_tensor_contract as contract

FOV_MM_XYZ = (420.0, 300.0, 380.0)

TARGET_SHAPE_CZYX = (1, 96, 96, 96)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def load_subject_row(project_root: Path, subject_id: str) -> tuple[pd.Series, Path]:
    manifests = sorted((project_root / "data" / "deidentified_release_candidate").glob("*/*/manifest.csv"))
    rows = []
    for manifest_path in manifests:
        df = pd.read_csv(manifest_path, dtype=str)
        hit = df[df["public_subject_id"].eq(subject_id)]
        if not hit.empty:
            rows.append((hit.iloc[0], manifest_path))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one manifest row for {subject_id}, found {len(rows)}")
    return rows[0]

def materialize_dicom_dir(subject_id: str, dicom_dir: Path, output_dir: Path) -> dict[str, object]:
    if not dicom_dir.exists():
        raise FileNotFoundError(dicom_dir)
    image, orientation_meta = contract.read_image_lps(dicom_dir)
    arr_hu = contract.sitk.GetArrayFromImage(image).astype(np.float32)
    body_mask = contract.compute_body_proxy(arr_hu)
    bbox = contract.physical_bbox_from_mask(image, body_mask)
    if bbox.get("body_proxy_status") != "PASS":
        raise RuntimeError(f"Body proxy failed for {subject_id}: {bbox}")
    center = (
        float(bbox["body_center_x_mm"]),
        float(bbox["body_center_y_mm"]),
        float(bbox["body_center_z_mm"]),
    )
    tensor, params = contract.resample_tensor(image, center, FOV_MM_XYZ, use_antialias=True)
    if tuple(tensor.shape) != TARGET_SHAPE_CZYX:
        raise RuntimeError(f"Tensor shape mismatch for {subject_id}: {tensor.shape}")
    tensor = tensor.astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{subject_id}_p95_96cubed_tensor.npy"
    np.save(tensor_path, tensor, allow_pickle=False)
    qc = {
        "public_subject_id": subject_id,
        "dicom_dir": str(dicom_dir),
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha256_file(tensor_path),
        "tensor_shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(np.min(tensor)),
        "max": float(np.max(tensor)),
        "mean": float(np.mean(tensor)),
        "std": float(np.std(tensor)),
        "fov_mm_xyz": list(FOV_MM_XYZ),
        "effective_spacing_xyz": [v / 96.0 for v in FOV_MM_XYZ],
        "hu_clip": [-1000.0, 400.0],
        "normalization": "fixed_minmax_-1000_to_400_to_0_1",
        "orientation_meta": orientation_meta,
        "body_proxy": bbox,
        "resampling_params": params,
    }
    qc_path = output_dir / f"{subject_id}_p95_96cubed_tensor_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    qc["qc_path"] = str(qc_path)
    return qc

def materialize_subject(project_root: Path, subject_id: str, output_dir: Path) -> dict[str, object]:
    row, manifest_path = load_subject_row(project_root, subject_id)
    subject_root = manifest_path.parent
    dicom_dir = subject_root / str(row["ct_path"])
    qc = materialize_dicom_dir(subject_id, dicom_dir, output_dir)
    qc["manifest_path"] = str(manifest_path)
    Path(qc["qc_path"]).write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    return qc

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate locked-contract whole-chest P95 96-cubed tensor from local DICOM.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[4]))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "p95_tensor_replay"
    report = materialize_subject(project_root, args.subject_id, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
