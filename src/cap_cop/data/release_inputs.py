# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import os

import json

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[3]))

DATA_ROOT = PROJECT_ROOT / "data" / "deidentified_release_candidate"

EXPECTED_COUNTS = {
    ("internal", "CAP"): 250,
    ("internal", "COP"): 125,
    ("external", "CAP"): 19,
    ("external", "COP"): 14,
}

def clinical_features() -> list[str]:
    config_root = Path(__file__).resolve().parents[3] / "configs"
    if not config_root.exists():
        config_root = Path(__file__).resolve().parents[1] / "configs"
    config = config_root / "clinical" / "full_structured_clinical.json"
    return list(json.loads(config.read_text(encoding="utf-8"))["features"])

def cohort_root(cohort: str, diagnosis: str) -> Path:
    if (cohort, diagnosis) not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown cohort/diagnosis: {cohort}/{diagnosis}")
    return DATA_ROOT / cohort / diagnosis

def load_clinical_cohort(cohort: str, *, legacy_id_alias: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    features = clinical_features()
    for diagnosis, label in (("CAP", 0), ("COP", 1)):
        root = cohort_root(cohort, diagnosis)
        manifest = pd.read_csv(root / "manifest.csv", dtype={"public_subject_id": str})
        frame = pd.read_csv(root / "clinical" / "clinical_features.csv", dtype={"public_subject_id": str})
        expected = EXPECTED_COUNTS[(cohort, diagnosis)]
        required = {"public_subject_id", "label_binary", *features}
        if len(frame) != expected or len(manifest) != expected:
            raise ValueError(f"Unexpected {cohort}/{diagnosis} cohort size")
        if required.difference(frame.columns):
            raise ValueError(f"Missing clinical columns: {sorted(required.difference(frame.columns))}")
        for name, table in (("clinical", frame), ("manifest", manifest)):
            ids = table["public_subject_id"]
            if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
                raise ValueError(f"Invalid {cohort}/{diagnosis} {name} IDs")
            if not table["label_binary"].astype(int).eq(label).all():
                raise ValueError(f"Incorrect {cohort}/{diagnosis} {name} labels")
        if set(frame["public_subject_id"]) != set(manifest["public_subject_id"]):
            raise ValueError(f"Clinical/manifest mismatch for {cohort}/{diagnosis}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["public_subject_id"].duplicated().any():
        raise ValueError(f"Duplicate {cohort} public IDs")
    if legacy_id_alias and cohort == "internal":
        # The locked fold manifests use this name for the same public case ID.
        combined["case_id_public"] = combined["public_subject_id"]
    return combined
