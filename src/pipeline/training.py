"""One scheduled retrain cycle for a pipeline (A or B), plus threshold
calibration on the validation split and pipeline persistence."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from src.drift.retrain_scheduler import decide, plan_cycle, should_train
from src.evaluation.metrics import threshold_for_fpr
from src.explainability import shap_audit
from src.features import feature_extractor as fx
from src.lsh.bucket_manager import HARD, BucketTable
from src.models.model_registry import ModelRegistry
from src.pipeline.inference import FALLBACK_ROUTES, ROUTE_CLASSIFIER, ROUTE_SIMPLE_MAL, DetectorPipeline

log = logging.getLogger(__name__)
FALLBACK_KEY = "__fallback__"


def benign_pool(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """§7.2: benign rows pooled from ALL buckets (capped for tractability)."""
    ben = np.flatnonzero(y == 0)
    if len(ben) > max_n:
        ben = np.sort(np.random.default_rng(seed).choice(ben, max_n, replace=False))
    return ben


def train_cycle(pmod, cfg, X: np.ndarray, y: np.ndarray, bids: np.ndarray, table: BucketTable,
                registry: ModelRegistry, now: float) -> dict:
    seed = cfg["sampling"]["random_seed"]
    cadence = cfg["retrain"]["cadence_days"]
    pool = benign_pool(y, cfg["models"]["benign_pool_max"], seed)
    hard = sorted(b for b, i in table.buckets.items() if i.state == HARD)
    last = {b: registry.last_trained_at(pmod.NAME, b) for b in hard}
    plan = plan_cycle(hard, set(table.trainable_buckets()), last, now, cadence)
    params = pmod.classifier_params(cfg)
    shap_reports = {}
    to_train = [b for b, a in plan.items() if should_train(a)]
    log.info("%s: %d hard buckets, training %d", pmod.NAME, len(hard), len(to_train))
    for k, b in enumerate(to_train, 1):
        in_bucket = bids == b
        mal = np.flatnonzero(in_bucket & (y == 1))
        t0 = time.time()
        model = pmod.train_classifier(X[mal], X[pool], params, seed=seed)
        registry.save(pmod.NAME, b, model, trained_at=now,
                      metadata={"n_malicious": int(len(mal)), "n_benign_pool": int(len(pool)),
                                "extractor_version": fx.EXTRACTOR_VERSION})
        if pmod.RUN_SHAP_AUDIT:
            sa = cfg["shap_audit"]
            shap_reports[b] = shap_audit.audit_bucket(model, X[np.flatnonzero(in_bucket)], sa["top_k"],
                                                      sa["watch_list"], seed=seed,
                                                      max_samples=sa["background_samples"])
        log.info("  [%d/%d] bucket %s: %d malicious + %d pooled benign (%.1fs)", k, len(to_train), b,
                 len(mal), len(pool), time.time() - t0)

    # §10.2 secondary-review model: always has enough data, same cadence.
    fb_action = decide(registry.last_trained_at(pmod.NAME, FALLBACK_KEY), True, now, cadence)
    if should_train(fb_action):
        idx = np.arange(len(y))
        cap = pmod.fallback_max_train_samples(cfg)
        if len(idx) > cap:
            idx = np.sort(np.random.default_rng(seed + 1).choice(idx, cap, replace=False))
        t0 = time.time()
        fb = pmod.train_classifier(X[idx[y[idx] == 1]], X[idx[y[idx] == 0]], pmod.fallback_params(cfg),
                                   seed=seed)
        registry.save(pmod.NAME, FALLBACK_KEY, fb, trained_at=now,
                      metadata={"n": int(len(idx)), "extractor_version": fx.EXTRACTOR_VERSION})
        log.info("%s: fallback trained on %d samples (%.1fs)", pmod.NAME, len(idx), time.time() - t0)

    shap_path = None
    if shap_reports:
        shap_path = shap_audit.write_cycle_report(cfg.path("shap_reports_dir"), pmod.NAME, now, shap_reports)
    return {"plan": {b: a.value for b, a in plan.items()}, "fallback": fb_action.value,
            "shap_report": str(shap_path) if shap_path else None,
            "shap_flagged": sorted(b for b, r in shap_reports.items() if r["flag_for_human_review"])}


def assemble(pmod, cfg, table: BucketTable, registry: ModelRegistry) -> DetectorPipeline:
    """Serve every HARD bucket's active model (including ones whose retrain was
    skipped this cycle, §8.2) plus the fallback."""
    classifiers = {}
    for b in registry.buckets(pmod.NAME):
        if b != FALLBACK_KEY and table.state(b) == HARD:
            classifiers[b] = registry.load_active(pmod.NAME, b)
    return DetectorPipeline(name=pmod.NAME, extractor_version=fx.EXTRACTOR_VERSION, lsh_cfg=dict(cfg["lsh"]),
                            buckets=table, classifiers=classifiers,
                            fallback=registry.load_active(pmod.NAME, FALLBACK_KEY))


def calibrate(pipe: DetectorPipeline, cfg, X_val: np.ndarray, E_val: np.ndarray, y_val: np.ndarray,
              bids_val: np.ndarray) -> dict:
    """Pick thresholds on the VALIDATION split only so that the full
    pipeline's FPR (simple-region lookups included) meets target_val_fpr."""
    c = cfg["calibration"]
    scored = pipe.score(E_val, bucket_ids=bids_val, features=X_val)
    s, r = scored["scores"], scored["routes"]
    ben = y_val == 0
    fp_simple = int((ben & (r == ROUTE_SIMPLE_MAL)).sum())
    allowed = int(np.floor(c["target_val_fpr"] * ben.sum())) - fp_simple
    model_routed = np.isin(r, (ROUTE_CLASSIFIER, *FALLBACK_ROUTES))
    t = threshold_for_fpr(s[ben & model_routed], allowed)
    pipe.threshold = t
    pipe.fallback_threshold = t * min(1.0, c["fallback_failsafe_factor"])
    return {"threshold": pipe.threshold, "fallback_threshold": pipe.fallback_threshold,
            "fp_from_simple_regions": fp_simple, "allowed_fp": allowed}


def save_pipeline(pipe: DetectorPipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path, compress=3)


def load_pipeline(path: Path) -> DetectorPipeline:
    pipe = joblib.load(path)
    fx.check_version(pipe.extractor_version)
    return pipe


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
