"""§9 SHAP audit, run after every successful hard-region RF (re)train.

For each retrained bucket: mean |SHAP| per feature (TreeExplainer) over a
sample of that bucket's own training rows, then the top-K features are
checked against a watch-list (resources / header features that ML models
over-rely on, that analysts trust least, and that simple evasion techniques
manipulate). A hit flags the bucket for HUMAN review; nothing is
auto-remediated.

One report per retrain cycle is written (never overwritten), so a reviewer
can tell a new flag from a persistent one.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.features.feature_extractor import FEATURE_NAMES


def _positive_class_shap(values) -> np.ndarray:
    v = values.values if hasattr(values, "values") else values
    if isinstance(v, list):          # older shap: one array per class
        return np.asarray(v[1])
    v = np.asarray(v)
    return v[..., 1] if v.ndim == 3 else v


def audit_bucket(model, X_background: np.ndarray, top_k: int, watch_list: list[str], seed: int = 0,
                 max_samples: int = 100) -> dict:
    import shap  # training-time dependency only

    rng = np.random.default_rng(seed)
    if len(X_background) > max_samples:
        X_background = X_background[rng.choice(len(X_background), max_samples, replace=False)]
    explainer = shap.TreeExplainer(model)
    sv = _positive_class_shap(explainer.shap_values(X_background, check_additivity=False))
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_k]
    top = [{"feature": FEATURE_NAMES[i], "mean_abs_shap": float(importance[i])} for i in order]
    present = set(FEATURE_NAMES)
    hits = [t["feature"] for t in top if t["feature"] in watch_list]
    return {
        "top_features": top,
        "watch_list_hits": hits,
        "flag_for_human_review": bool(hits),
        "watch_list_not_in_schema": [w for w in watch_list if w not in present],
    }


def write_cycle_report(out_dir: Path, pipeline: str, cycle_time: float, bucket_reports: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(cycle_time, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"shap_audit_{pipeline}_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    flagged = sorted(b for b, r in bucket_reports.items() if r.get("flag_for_human_review"))
    path.write_text(json.dumps({"pipeline": pipeline, "cycle_time_utc": stamp, "flagged_buckets": flagged,
                                "buckets": bucket_reports}, indent=1))
    return path
