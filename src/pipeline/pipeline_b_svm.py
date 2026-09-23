"""Pipeline B: identical LSH pre-filter -> per-bucket calibrated Linear SVM
(baseline that tests whether the Random Forest earns its complexity)."""
from src.models.baseline_svm import train_svm

NAME = "B_svm"
RUN_SHAP_AUDIT = False  # §9: the audit covers the production (RF) candidate only


def classifier_params(cfg) -> dict:
    return dict(cfg["models"]["linear_svm"])


def fallback_params(cfg) -> dict:
    return dict(cfg["models"]["linear_svm"])


train_classifier = train_svm
