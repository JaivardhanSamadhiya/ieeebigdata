#!/usr/bin/env python3
"""Verify that key reported values are present in the compact result package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    final = json.loads((DATA / "FINAL_MECHANISTIC_RESULT.json").read_text())
    hardening = json.loads((DATA / "ROBUSTNESS_AUDIT_RESULT.json").read_text())
    recovery = json.loads((DATA / "FINAL_VALIDATION_RESULT.json").read_text())
    metrics = pd.read_csv(DATA / "evaluation_design_metrics.csv")

    macro = metrics.groupby("design")[["spearman", "r2"]].mean()
    checks = {
        "random_macro_spearman": round(float(macro.loc["random_kfold", "spearman"]), 3),
        "complete_cc_macro_spearman": round(float(macro.loc["complete_cc_holdout", "spearman"]), 3),
        "complete_cc_macro_r2": round(float(macro.loc["complete_cc_holdout", "r2"]), 3),
        "lineage_eta2_macro": round(float(final["lineage_eta2_macro"]), 3),
        "same_cc_auc": round(float(hardening["genome_similarity_audit"]["same_cc_auc_from_jaccard"]), 3),
        "recovery_balanced_accuracy": round(float(recovery["pooled_metrics"]["balanced_accuracy"]), 3),
        "recovery_mcc": round(float(recovery["pooled_metrics"]["MCC"]), 3),
    }
    expected = {
        "random_macro_spearman": 0.457,
        "complete_cc_macro_spearman": 0.190,
        "complete_cc_macro_r2": -0.237,
        "lineage_eta2_macro": 0.235,
        "same_cc_auc": 0.994,
        "recovery_balanced_accuracy": 0.492,
        "recovery_mcc": -0.024,
    }
    assert checks == expected, {"observed": checks, "expected": expected}
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
