"""§10 full-pipeline routing.

For each sample:
  1. EMBER v2 vector -> model features (versioned extractor); bucket via the
     fitted LSH mapping in query mode (never re-fit).
  2. simple_malicious bucket -> malicious (lookup only).
  3. simple_benign bucket    -> benign (lookup only).
  4. hard bucket with a trained classifier -> that classifier.
  5. hard bucket without a classifier (§10.2) or a bucket never seen in
     training (§10.3) -> fail-safe route: the "secondary review" model, a
     classifier trained on the whole training split, whose threshold can be
     biased toward malicious by ``fallback_failsafe_factor``. Neither case is
     ever auto-passed as benign.

The same class serves Pipeline A (Random Forest) and Pipeline B (Linear SVM);
they share the LSH mapping and bucket table and differ only in classifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.features import feature_extractor as fx
from src.lsh.bucket_manager import HARD, NOVEL, SIMPLE_BENIGN, SIMPLE_MALICIOUS, BucketTable
from src.lsh.hasher import LSHPrefilter

ROUTE_SIMPLE_MAL = "simple_malicious"
ROUTE_SIMPLE_BEN = "simple_benign"
ROUTE_CLASSIFIER = "bucket_classifier"
ROUTE_FALLBACK_HARD = "fallback_hard_untrained"
ROUTE_FALLBACK_NOVEL = "fallback_novel_bucket"
FALLBACK_ROUTES = (ROUTE_FALLBACK_HARD, ROUTE_FALLBACK_NOVEL)


@dataclass
class DetectorPipeline:
    name: str
    extractor_version: str
    lsh_cfg: dict
    buckets: BucketTable
    classifiers: dict = field(default_factory=dict)   # bucket id -> model with predict_proba
    fallback: object = None
    threshold: float = 0.5
    fallback_threshold: float = 0.5

    def __post_init__(self):
        fx.check_version(self.extractor_version)
        self.lsh = LSHPrefilter(self.lsh_cfg)

    # -- routing -------------------------------------------------------------
    def route(self, ember: np.ndarray, bucket_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        E = np.atleast_2d(ember)
        bids = self.lsh.buckets(E) if bucket_ids is None else bucket_ids
        routes = np.empty(len(E), dtype=object)
        for i, b in enumerate(bids):
            state = self.buckets.state(b)
            if state == SIMPLE_MALICIOUS:
                routes[i] = ROUTE_SIMPLE_MAL
            elif state == SIMPLE_BENIGN:
                routes[i] = ROUTE_SIMPLE_BEN
            elif state == HARD and b in self.classifiers:
                routes[i] = ROUTE_CLASSIFIER
            elif state == NOVEL:
                routes[i] = ROUTE_FALLBACK_NOVEL
            else:
                routes[i] = ROUTE_FALLBACK_HARD
        return bids, routes

    def score(self, ember: np.ndarray, bucket_ids: np.ndarray | None = None,
              features: np.ndarray | None = None) -> dict:
        """Malicious scores + routing info. Simple regions score 1.0 / 0.0."""
        E = np.atleast_2d(ember)
        bids, routes = self.route(E, bucket_ids)
        X = fx.transform(E) if features is None else features
        scores = np.zeros(len(E), dtype=np.float64)
        scores[routes == ROUTE_SIMPLE_MAL] = 1.0
        clf_rows = np.flatnonzero(routes == ROUTE_CLASSIFIER)
        for b in np.unique(bids[clf_rows]) if len(clf_rows) else []:
            rows = clf_rows[bids[clf_rows] == b]
            scores[rows] = self.classifiers[b].predict_proba(X[rows])[:, 1]
        fb_rows = np.flatnonzero(np.isin(routes, FALLBACK_ROUTES))
        if len(fb_rows):
            scores[fb_rows] = self.fallback.predict_proba(X[fb_rows])[:, 1]
        return {"scores": scores, "routes": routes, "buckets": bids}

    def decide(self, scored: dict) -> np.ndarray:
        s, r = scored["scores"], scored["routes"]
        verdict = np.zeros(len(s), dtype=np.int8)
        verdict[r == ROUTE_SIMPLE_MAL] = 1
        clf = r == ROUTE_CLASSIFIER
        verdict[clf] = s[clf] >= self.threshold
        fb = np.isin(r, FALLBACK_ROUTES)
        verdict[fb] = s[fb] >= self.fallback_threshold
        return verdict

    def predict(self, ember: np.ndarray) -> np.ndarray:
        return self.decide(self.score(ember))
