"""§11 final Pipeline A (LSH+RF) vs Pipeline B (LSH+SVM) comparison.

Both pipelines are evaluated as FULL pipelines (simple-region lookups +
bucket classifiers + fallback), with the identical LSH mapping and bucket
table, on the identical temporal TEST split.

Test-set discipline (§11.4): the test split is meant to be touched exactly
once. This script records that it ran and refuses to run again unless
--allow-test-reuse is given (all tuning must use the validation split).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.data.dataset import load_features, load_split  # noqa: E402
from src.evaluation.metrics import report  # noqa: E402
from src.pipeline import training  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipelines", nargs="+", default=["A_rf", "B_svm"])
    ap.add_argument("--allow-test-reuse", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg.path("eval_reports_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "TEST_SPLIT_USED.json"
    if marker.exists() and not args.allow_test_reuse:
        raise SystemExit(f"the test split was already evaluated ({marker.read_text().strip()}); "
                         "tune on validation instead, or pass --allow-test-reuse knowingly")

    df, E = load_split(cfg, "test")
    y = df["y"].to_numpy()
    X = load_features(cfg, "test", E)
    pipes = {n: training.load_pipeline(cfg.path("processed_dir") / "pipelines" / f"{n}.joblib")
             for n in args.pipelines}
    first = next(iter(pipes.values()))
    bids = first.lsh.buckets(E)          # identical mapping for every pipeline
    results = {}
    for name, pipe in pipes.items():
        if pipe.lsh_cfg != first.lsh_cfg or pipe.buckets.to_dict() != first.buckets.to_dict():
            raise SystemExit(f"{name} does not share the LSH mapping/bucket table; comparison would be unfair")
        scored = pipe.score(E, bucket_ids=bids, features=X)
        pred = pipe.decide(scored)
        per_route = {}
        for r in np.unique(scored["routes"]):
            m = scored["routes"] == r
            per_route[r] = {"n": int(m.sum()), **{k: v for k, v in report(y[m], pred[m]).items()
                                                  if k in ("tp", "fp", "tn", "fn")}}
        results[name] = {"full_pipeline": report(y, pred), "by_route": per_route,
                         "threshold": pipe.threshold, "fallback_threshold": pipe.fallback_threshold}
        fp = results[name]["full_pipeline"]
        logging.info("%s TEST: TPR %.4f FPR %.4f precision %.4f MCC %.4f | %s | %s", name, fp["recall_tpr"],
                     fp["fpr"], fp["precision"], fp["mcc"], fp["false_positives"], fp["false_negatives"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out_dir / f"test_comparison_{stamp}.json"
    path.write_text(json.dumps({"n_test": int(len(y)), "n_benign": int((y == 0).sum()),
                                "n_malicious": int((y == 1).sum()), "results": results}, indent=1))
    marker.write_text(json.dumps({"evaluated_at": stamp, "report": path.name}))
    logging.info("report -> %s", path)


if __name__ == "__main__":
    main()
