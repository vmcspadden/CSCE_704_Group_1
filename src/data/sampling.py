"""§3.3 sampling: two separate mechanisms.

1. Confounding avoidance (:func:`confounding_report`): a diagnostic that flags
   any data source supplying more than ``max_single_source_fraction`` of a
   class. It only reports; fixing an over-concentrated source (sourcing more
   diversely vs. capping it) is a human decision.

2. Realistic prevalence (:func:`target_prevalence`): brings an evaluation
   split to a benign-dominated benign:malicious ratio by stratified (by
   source) downsampling of the benign class. Malicious samples are never
   oversampled or synthesized.

:func:`budget_subsample` is a third, unrelated tool: uniform random
subsampling of a class to fit a compute budget (download / training size).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def confounding_report(df: pd.DataFrame, max_single_source_fraction: float) -> dict:
    """Per-class source concentration. ``flagged`` lists (class, source) pairs
    that exceed the threshold and need human review."""
    report: dict = {"max_single_source_fraction": max_single_source_fraction, "classes": {}, "flagged": []}
    for label, part in df.groupby("label"):
        fractions = part["source"].value_counts(normalize=True)
        report["classes"][label] = {"n": int(len(part)), "source_fractions": fractions.round(4).to_dict()}
        for source, frac in fractions.items():
            if frac > max_single_source_fraction:
                report["flagged"].append({"label": label, "source": source, "fraction": round(float(frac), 4)})
    for f in report["flagged"]:
        log.warning("confounding check: source %r supplies %.1f%% of %s samples - needs human review",
                     f["source"], 100 * f["fraction"], f["label"])
    return report


def target_prevalence(df: pd.DataFrame, benign_to_malicious_ratio: float, seed: int) -> pd.DataFrame:
    """Downsample BENIGN rows (stratified by source) so that
    n_benign ~= ratio * n_malicious. Never touches malicious rows."""
    mal = df[df["label"] == "MALICIOUS"]
    ben = df[df["label"] == "BENIGN"]
    target = int(round(benign_to_malicious_ratio * len(mal)))
    if len(ben) <= target:
        log.warning("only %d benign samples for a target of %d; prevalence left at %.2f:1 "
                    "(malicious data is never oversampled)", len(ben), target,
                    len(ben) / max(1, len(mal)))
        return df
    frac = target / len(ben)
    rng = np.random.default_rng(seed)
    keep = []
    for _, part in ben.groupby("source"):
        n = int(round(frac * len(part)))
        keep.append(part.iloc[np.sort(rng.choice(len(part), size=n, replace=False))])
    kept = pd.concat([mal, *keep])
    return kept.sort_values("first_seen_utc", kind="mergesort").reset_index(drop=True)


def budget_subsample(df: pd.DataFrame, max_per_label: dict[str, int], seed: int) -> pd.DataFrame:
    """Uniform random cap per label (a compute budget, not a prevalence tool)."""
    rng = np.random.default_rng(seed)
    parts = []
    for label, part in df.groupby("label"):
        cap = max_per_label.get(label)
        if cap is not None and len(part) > cap:
            part = part.iloc[np.sort(rng.choice(len(part), size=cap, replace=False))]
        parts.append(part)
    return pd.concat(parts).sort_values("first_seen_utc", kind="mergesort").reset_index(drop=True)
