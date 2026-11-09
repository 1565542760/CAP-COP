# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import pandas as pd

from cap_cop.paths import project_root

SPLIT_SEED = "20260627"

@dataclass(frozen=True)
class FoldMembership:
    outer_fold: int
    inner_fold: int
    train_ids: list[str]
    validation_ids: list[str]
    outer_test_ids: list[str]

def split_manifest_path() -> Path:
    return project_root() / "models" / "segmentation" / "manifests" / "locked_nested_split_patient_manifest.csv"

def load_locked_split_manifest(path: str | Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or split_manifest_path(), dtype=str)
    return frame[frame["training_seed"].eq(SPLIT_SEED)].copy()

def fold_membership(outer_fold: int, inner_fold: int, split_frame: pd.DataFrame | None = None) -> FoldMembership:
    frame = load_locked_split_manifest() if split_frame is None else split_frame.copy()
    outer = frame[frame["outer_fold_id"].eq(str(outer_fold))]
    outer_test = sorted(outer[outer["outer_role"].eq("outer_test")]["patient_group_id_public"].tolist())
    outer_train = outer[outer["outer_role"].eq("outer_train")]
    validation = sorted(outer_train[outer_train["inner_fold_id"].eq(str(inner_fold))]["patient_group_id_public"].tolist())
    train = sorted(set(outer_train["patient_group_id_public"].tolist()) - set(validation))
    return FoldMembership(outer_fold, inner_fold, train, validation, outer_test)

def assert_strict_oof_patient_not_in_training(patient_group_id_public: str, outer_fold: int) -> None:
    split_frame = load_locked_split_manifest()
    memberships = [fold_membership(outer_fold, inner, split_frame) for inner in range(4)]
    if not all(patient_group_id_public in item.outer_test_ids for item in memberships):
        raise ValueError(f"{patient_group_id_public} is not in outer-test fold {outer_fold}.")
    leaking = [item.inner_fold for item in memberships if patient_group_id_public in item.train_ids or patient_group_id_public in item.validation_ids]
    if leaking:
        raise ValueError(f"Strict OOF leakage: {patient_group_id_public} appears in fold(s) {leaking}.")

def verify_all_strict_oof_folds() -> dict[str, object]:
    split_frame = load_locked_split_manifest()
    checked = 0
    for outer in range(5):
        for inner in range(4):
            membership = fold_membership(outer, inner, split_frame)
            if set(membership.train_ids) & set(membership.validation_ids):
                raise ValueError(f"Train/validation overlap in outer {outer}, inner {inner}.")
            if set(membership.outer_test_ids) & set(membership.train_ids):
                raise ValueError(f"Outer-test/train overlap in outer {outer}, inner {inner}.")
            if set(membership.outer_test_ids) & set(membership.validation_ids):
                raise ValueError(f"Outer-test/validation overlap in outer {outer}, inner {inner}.")
            checked += 1
    return {"status": "passed", "checked_outer_inner_runs": checked}
