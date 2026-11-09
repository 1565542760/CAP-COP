"""Refit A's published final-model algorithm on the locked internal feature table.

This phase never reads external labels or performance; the output is label-blind.
"""

from __future__ import annotations

import argparse

import hashlib

import json

import platform

import shutil

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

import sklearn

import run_isolated_a_roi as run_a

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final A refit: {output}")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "run_source_snapshot.py")
    _, labels, matrix, x, y, folds, _ = run_a.load_inputs()
    if len(x) != 375 or x.index.nunique() != 375 or set(y) != {0, 1}:
        raise ValueError("A final-fit internal input contract failed")
    source = run_a.a
    source.DIRS["robust"] = output / "robust"
    source.DIRS["final"] = output / "final"
    manifest = {"status": "INTERNAL_FIT_STARTED_EXTERNAL_LABELS_NOT_READ", "input_sha256": {
        str(run_a.MATRIX_PATH): digest(run_a.MATRIX_PATH),
        str(run_a.XWALK_PATH): digest(run_a.XWALK_PATH),
        str(Path(source.__file__)): digest(Path(source.__file__)),
        str(Path(run_a.__file__)): digest(Path(run_a.__file__)),
        str(Path(__file__)): digest(Path(__file__)),
    }, "internal_patient_count": 375, "internal_fold_count": int(folds.nunique()),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "external_labels_read": False}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_path, lock, features = source.fit_final(labels, matrix)
    if digest(model_path) != lock["model_sha256"]:
        raise ValueError("A final model hash differs from its pre-external lock")
    ext_path = run_a.PROJECT / "models/lesion_radiomics/external_label_blind_pyradiomics_features.csv"
    ext = pd.read_csv(ext_path)
    if len(ext) != 33 or ext.neutral_patient_id.nunique() != 33:
        raise ValueError("A external label-blind feature table does not cover 33 unique patients")
    missing = sorted(set(features) - set(ext.columns))
    if missing:
        manifest.update({"status": "EXTERNAL_INPUT_FEATURES_MISSING", "missing_features": missing,
                         "final_model_sha256": digest(model_path), "external_labels_read": False})
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise ValueError(f"A final refit needs unavailable external feature columns: {missing}")
    fitted = joblib.load(model_path)
    p = fitted["pipeline"].predict_proba(ext[features])[:, 1]
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("A final model produced invalid external probabilities")
    predictions = output / "A_external_label_blind_refit_predictions.csv"
    pd.DataFrame({"neutral_patient_id": ext.neutral_patient_id, "probability_COP": p}).to_csv(predictions, index=False)
    manifest.update({"status": "A_FINAL_MODEL_REFIT_AND_EXTERNAL_LABEL_BLIND_PREDICTED",
                     "external_features_sha256": digest(ext_path), "final_model_sha256": digest(model_path),
                     "external_predictions_sha256": digest(predictions), "selected_features": features,
                     "selected_model": lock["model"], "selected_params": lock["params"],
                     "external_labels_read": False})
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "n_features": len(features),
                      "model": lock["model"], "external_patients": len(ext)}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(Path(args.output).resolve())
