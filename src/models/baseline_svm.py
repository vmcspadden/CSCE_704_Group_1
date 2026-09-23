"""§7.1/7.5 Pipeline B hard-region classifier: calibrated linear SVM baseline.

The model is a linear SVM (hinge loss, L2 penalty, class_weight='balanced')
wrapped in CalibratedClassifierCV so that it exposes predict_proba like the
Random Forest, letting thresholding and metrics code treat both pipelines
identically. Features are standardized first (the feature extractor already
applies a signed log1p). Calibration folds are internal to
CalibratedClassifierCV on training data - this is not k-fold evaluation.

Implementation note: the spec suggests sklearn's LinearSVC. On these 2394
dense features liblinear needed ~460 s for a single 42k-sample fit even at
tol=1e-3 (the 300k-sample fallback would take hours), so the same model
(hinge-loss linear SVM) is trained with SGDClassifier instead, which fits the
300k fallback in ~2.5 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_svm(params: dict, seed: int = 0):
    svm = SGDClassifier(loss="hinge", penalty="l2", alpha=params.get("alpha", 1e-5),
                        class_weight=params.get("class_weight", "balanced"),
                        max_iter=params.get("max_iter", 50), tol=params.get("tol", 1e-4), random_state=seed)
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, cv=params.get("calibration_cv", 3)))


def train_svm(X_mal: np.ndarray, X_ben: np.ndarray, params: dict, seed: int = 0):
    X = np.vstack([X_ben, X_mal])
    y = np.r_[np.zeros(len(X_ben), dtype=np.int8), np.ones(len(X_mal), dtype=np.int8)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return build_svm(params, seed).fit(X, y)
