"""§3.1 AVClass-style multi-engine consensus labeling.

A single vendor's verdict is never trusted. Each sample is labeled from the
number of engines that scanned it and the number that flagged it:

  engines_reporting < min_engines_reporting          -> INSUFFICIENT_COVERAGE
  engines_malicious >= malicious_consensus_threshold -> MALICIOUS
  engines_malicious <= benign_consensus_threshold    -> BENIGN
  anything in between                                -> AMBIGUOUS

INSUFFICIENT_COVERAGE and AMBIGUOUS samples are excluded from training and
evaluation. Samples whose consensus was computed fewer than
relabel_window_days after first_seen are flagged for re-verification.
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0


class Label(str, Enum):
    BENIGN = "BENIGN"
    MALICIOUS = "MALICIOUS"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"


TRAINABLE = (Label.BENIGN.value, Label.MALICIOUS.value)


def consensus_label(engines_malicious: float, engines_reporting: float | None, cfg) -> Label:
    if cfg["malicious_consensus_threshold"] <= cfg["benign_consensus_threshold"]:
        raise ValueError("malicious_consensus_threshold must exceed benign_consensus_threshold")
    if engines_reporting is not None and not pd.isna(engines_reporting):
        if engines_reporting < cfg["min_engines_reporting"]:
            return Label.INSUFFICIENT_COVERAGE
    if engines_malicious is None or pd.isna(engines_malicious):
        return Label.INSUFFICIENT_COVERAGE
    if engines_malicious >= cfg["malicious_consensus_threshold"]:
        return Label.MALICIOUS
    if engines_malicious <= cfg["benign_consensus_threshold"]:
        return Label.BENIGN
    return Label.AMBIGUOUS


def apply_labels(manifest: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add ``label``, ``y`` and ``needs_reverification`` columns.

    Expects columns: engines_malicious, engines_reporting (may be all-NaN when
    the source does not publish scan coverage), first_seen_utc and
    consensus_computed_at_utc (epoch seconds).
    """
    df = manifest.copy()
    mal_t, ben_t = cfg["malicious_consensus_threshold"], cfg["benign_consensus_threshold"]
    if mal_t <= ben_t:
        raise ValueError("malicious_consensus_threshold must exceed benign_consensus_threshold")

    reporting = pd.to_numeric(df.get("engines_reporting"), errors="coerce") \
        if "engines_reporting" in df else pd.Series(np.nan, index=df.index)
    positives = pd.to_numeric(df["engines_malicious"], errors="coerce")
    if reporting.isna().all():
        log.warning("engines_reporting unavailable for every sample; coverage check skipped")

    label = np.full(len(df), Label.AMBIGUOUS.value, dtype=object)
    label[(positives >= mal_t).to_numpy()] = Label.MALICIOUS.value
    label[(positives <= ben_t).to_numpy()] = Label.BENIGN.value
    low_coverage = (reporting.notna() & (reporting < cfg["min_engines_reporting"])) | positives.isna()
    label[low_coverage.to_numpy()] = Label.INSUFFICIENT_COVERAGE.value
    df["label"] = label
    df["y"] = (df["label"] == Label.MALICIOUS.value).astype(np.int8)

    age_days = (pd.to_numeric(df["consensus_computed_at_utc"]) - pd.to_numeric(df["first_seen_utc"])) \
        / SECONDS_PER_DAY
    df["needs_reverification"] = age_days < cfg["relabel_window_days"]
    return df


def trainable(df: pd.DataFrame, allow_unverified: bool = False) -> pd.DataFrame:
    """Keep BENIGN/MALICIOUS samples; drop those awaiting re-verification."""
    keep = df["label"].isin(TRAINABLE)
    if not allow_unverified:
        keep &= ~df["needs_reverification"]
    return df[keep]


def label_summary(df: pd.DataFrame) -> dict:
    return {
        "counts": df["label"].value_counts().to_dict(),
        "needs_reverification": int(df["needs_reverification"].sum()),
    }
