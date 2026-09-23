"""§7.1/7.4 Pipeline A hard-region classifier: Random Forest.

Per bucket: malicious samples from that bucket only, benign samples pooled
from all buckets (§7.2). Imbalance is handled per bucket with
class_weight='balanced_subsample' (§7.3), not by resampling.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_rf(params: dict, seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        class_weight=params.get("class_weight", "balanced_subsample"),
        n_jobs=params.get("n_jobs", -1),
        random_state=seed,
    )


def train_rf(X_mal: np.ndarray, X_ben: np.ndarray, params: dict, seed: int = 0) -> RandomForestClassifier:
    X = np.vstack([X_ben, X_mal])
    y = np.r_[np.zeros(len(X_ben), dtype=np.int8), np.ones(len(X_mal), dtype=np.int8)]
    model = build_rf(params, seed).fit(X, y)
    model.set_params(n_jobs=1)  # single-threaded at inference (one CPU in the container)
    return model
