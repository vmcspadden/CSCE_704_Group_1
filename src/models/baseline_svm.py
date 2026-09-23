"""§7.1/7.5 Pipeline B hard-region classifier: calibrated Linear SVM baseline.

LinearSVC is wrapped in CalibratedClassifierCV so that it exposes
predict_proba like the Random Forest, letting thresholding and metrics code
treat both pipelines identically. Features are standardized first (the
feature extractor already applies a signed log1p). Calibration folds are
internal to CalibratedClassifierCV on the training data - this is not
k-fold model evaluation.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_svm(params: dict, seed: int = 0):
    svc = LinearSVC(C=params.get("C", 0.1), class_weight=params.get("class_weight", "balanced"),
                    max_iter=params.get("max_iter", 5000), dual="auto", random_state=seed)
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, cv=params.get("calibration_cv", 3)))


def train_svm(X_mal: np.ndarray, X_ben: np.ndarray, params: dict, seed: int = 0):
    X = np.vstack([X_ben, X_mal])
    y = np.r_[np.zeros(len(X_ben), dtype=np.int8), np.ones(len(X_mal), dtype=np.int8)]
    return build_svm(params, seed).fit(X, y)
