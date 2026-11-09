# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import argparse

import json

from pathlib import Path

import numpy as np

import torch

from cap_cop.imaging.segmentation.inference import THRESHOLD, checkpoint_paths_for_outer, ensemble_probability, mask_from_probability

from cap_cop.imaging.segmentation.preprocess import preprocess_dicom_for_segmentation

from cap_cop.imaging.segmentation.splits import assert_strict_oof_patient_not_in_training

from cap_cop.paths import project_root, resolve_project_path

def predict_internal_strict_oof(
    public_subject_id: str,
    patient_group_id_public: str,
    outer_fold: int,
    dicom_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "cpu",
    max_models: int | None = None,
) -> dict[str, object]:
    assert_strict_oof_patient_not_in_training(patient_group_id_public, outer_fold)
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    preprocessed = preprocess_dicom_for_segmentation(dicom_dir)
    checkpoint_paths = checkpoint_paths_for_outer(outer_fold)
    probability = ensemble_probability(preprocessed.array_zyx, checkpoint_paths, device, max_models=max_models)
    mask = mask_from_probability(probability, THRESHOLD)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / f"{public_subject_id}_strict_oof_predicted_mask.npy"
    prob_path = out / f"{public_subject_id}_strict_oof_probability.npy"
    np.save(mask_path, mask)
    np.save(prob_path, probability.astype(np.float32))
    summary = {
        "public_subject_id": public_subject_id,
        "patient_group_id_public": patient_group_id_public,
        "outer_fold": outer_fold,
        "models_used": len(checkpoint_paths[:max_models] if max_models else checkpoint_paths),
        "locked_models_expected_for_full_run": 4,
        "threshold": THRESHOLD,
        "shape_zyx": list(mask.shape),
        "spacing_xyz": list(preprocessed.spacing_xyz),
        "mask_voxels": int(mask.sum()),
        "mask_nonempty": bool(mask.sum() > 0),
        "mask_path": str(mask_path),
        "probability_path": str(prob_path),
    }
    (out / f"{public_subject_id}_strict_oof_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked strict-OOF lesion segmentation for one internal case.")
    parser.add_argument("--public-subject-id", required=True)
    parser.add_argument("--patient-group-id-public", required=True)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("--dicom-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/segmentation/internal")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-models", type=int, default=None, help="Smoke-test limiter; omit for the locked 4-model outer-fold ensemble.")
    args = parser.parse_args()
    dicom_dir = resolve_project_path(args.dicom_dir) if not Path(args.dicom_dir).is_absolute() else Path(args.dicom_dir)
    output_dir = resolve_project_path(args.output_dir)
    summary = predict_internal_strict_oof(
        args.public_subject_id,
        args.patient_group_id_public,
        args.outer_fold,
        dicom_dir,
        output_dir,
        args.device,
        args.max_models,
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
