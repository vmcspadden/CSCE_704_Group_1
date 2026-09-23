"""Pipeline A: LSH pre-filter -> per-bucket Random Forest (production candidate)."""
from src.models.hard_region_rf import train_rf

NAME = "A_rf"
RUN_SHAP_AUDIT = True


def classifier_params(cfg) -> dict:
    return dict(cfg["models"]["random_forest"])


def fallback_params(cfg) -> dict:
    return dict(cfg["models"]["fallback"]["random_forest"])


train_classifier = train_rf
