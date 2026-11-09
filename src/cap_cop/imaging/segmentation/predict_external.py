# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import argparse

import json

from pathlib import Path

import numpy as np

import torch

from cap_cop.imaging.segmentation.inference import THRESHOLD, all_checkpoint_paths, ensemble_probability, mask_from_probability

from cap_cop.imaging.segmentation.preprocess import preprocess_dicom_for_segmentation

from cap_cop.paths import resolve_project_path

def predict_external_ensemble(
    public_subject_id: str,
    dicom_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "cpu",
    max_models: int | None = None,
) -> dict[str, object]:
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    preprocessed = preprocess_dicom_for_segmentation(dicom_dir)
    checkpoint_paths = all_checkpoint_paths()
    probability = ensemble_probability(preprocessed.array_zyx, checkpoint_paths, device, max_models=max_models)
    mask = mask_from_probability(probability, THRESHOLD)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / f"{public_subject_id}_external_predicted_mask.npy"
    prob_path = out / f"{public_subject_id}_external_probability.npy"
    np.save(mask_path, mask)
    np.save(prob_path, probability.astype(np.float32))
    summary = {
        "public_subject_id": public_subject_id,
        "models_used": len(checkpoint_paths[:max_models] if max_models else checkpoint_paths),
        "locked_models_expected_for_full_run": 20,
        "threshold": THRESHOLD,
        "shape_zyx": list(mask.shape),
        "spacing_xyz": list(preprocessed.spacing_xyz),
        "mask_voxels": int(mask.sum()),
        "mask_nonempty": bool(mask.sum() > 0),
        "mask_path": str(mask_path),
        "probability_path": str(prob_path),
    }
    (out / f"{public_subject_id}_external_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked 20-checkpoint lesion segmentation ensemble for one external case.")
    parser.add_argument("--public-subject-id", required=True)
    parser.add_argument("--dicom-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/segmentation/external")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-models", type=int, default=None, help="Smoke-test limiter; omit for the locked 20-model ensemble.")
    args = parser.parse_args()
    dicom_dir = resolve_project_path(args.dicom_dir) if not Path(args.dicom_dir).is_absolute() else Path(args.dicom_dir)
    output_dir = resolve_project_path(args.output_dir)
    summary = predict_external_ensemble(args.public_subject_id, dicom_dir, output_dir, args.device, args.max_models)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
