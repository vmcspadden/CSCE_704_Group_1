"""§6 bucket state management (two states, as in Kaspersky's design).

  SIMPLE: every training sample in the bucket has one class -> the hash
          lookup alone decides (simple_malicious / simple_benign).
  HARD:   the bucket mixes classes -> routed to that bucket's classifier.

Implementation decisions, recorded here and in the README:
  * A single-class bucket must hold at least ``simple_region_min_samples``
    training samples before a lookup alone is trusted. Below that it is a
    HARD region (there is too little evidence for purity) - it gets no
    classifier, since it cannot meet the per-bucket minimums, and is routed
    per §10.2. This is not a third state; it is how "hard" is defined for
    tiny buckets.
  * Buckets are built from training-split samples only.

A HARD bucket is *trainable* when it has at least
``hard_region_min_malicious_samples`` malicious samples of its own and the
pooled benign set (all buckets, §7.2) holds at least
``hard_region_min_benign_samples``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIMPLE_MALICIOUS = "simple_malicious"
SIMPLE_BENIGN = "simple_benign"
HARD = "hard"
NOVEL = "novel"  # query-time only: bucket id never seen in training


@dataclass
class BucketInfo:
    n_benign: int
    n_malicious: int
    state: str


class BucketTable:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buckets: dict[str, BucketInfo] = {}
        self.pooled_benign = 0

    def fit(self, bucket_ids: np.ndarray, y: np.ndarray) -> "BucketTable":
        """Build states from TRAINING samples only (never val/test)."""
        import pandas as pd  # training-time only; the serving image has no pandas

        df = pd.DataFrame({"b": bucket_ids, "y": np.asarray(y).astype(int)})
        counts = df.groupby("b")["y"].agg(["size", "sum"])
        self.buckets = {}
        for b, row in counts.iterrows():
            n_mal = int(row["sum"])
            n_ben = int(row["size"]) - n_mal
            self.buckets[b] = BucketInfo(n_ben, n_mal, self._state(n_ben, n_mal))
        self.pooled_benign = int((df["y"] == 0).sum())
        return self

    def _state(self, n_ben: int, n_mal: int) -> str:
        if n_ben + n_mal >= self.cfg["simple_region_min_samples"]:
            if n_ben == 0:
                return SIMPLE_MALICIOUS
            if n_mal == 0:
                return SIMPLE_BENIGN
        return HARD

    def state(self, bucket_id: str) -> str:
        info = self.buckets.get(bucket_id)
        return NOVEL if info is None else info.state

    def trainable(self, bucket_id: str) -> bool:
        info = self.buckets.get(bucket_id)
        return (info is not None and info.state == HARD
                and info.n_malicious >= self.cfg["hard_region_min_malicious_samples"]
                and self.pooled_benign >= self.cfg["hard_region_min_benign_samples"])

    def trainable_buckets(self) -> list[str]:
        return sorted(b for b in self.buckets if self.trainable(b))

    def summary(self) -> dict:
        import pandas as pd

        states = pd.Series({b: i.state for b, i in self.buckets.items()}, dtype=object)
        sizes = pd.Series({b: i.n_benign + i.n_malicious for b, i in self.buckets.items()})
        return {
            "n_buckets": len(self.buckets),
            "buckets_by_state": states.value_counts().to_dict(),
            "train_samples_by_state": sizes.groupby(states).sum().to_dict(),
            "trainable_hard_buckets": len(self.trainable_buckets()),
            "pooled_benign": self.pooled_benign,
        }

    def to_dict(self) -> dict:
        return {"pooled_benign": self.pooled_benign,
                "buckets": {b: [i.n_benign, i.n_malicious, i.state] for b, i in self.buckets.items()}}

    @classmethod
    def from_dict(cls, cfg, d: dict) -> "BucketTable":
        t = cls(cfg)
        t.pooled_benign = d["pooled_benign"]
        t.buckets = {b: BucketInfo(v[0], v[1], v[2]) for b, v in d["buckets"].items()}
        return t
