"""Run one retrain cycle (§8) for Pipeline A (RF) and/or B (SVM).

  1. load the training split; features via the versioned extractor
  2. LSH buckets from the fitted (seeded) mapping; bucket table built from
     TRAINING samples only
  3. per-bucket gating via the retrain scheduler; train / retrain eligible
     hard buckets; SHAP audit after every RF retrain; fallback model
  4. calibrate thresholds on the VALIDATION split (never test)
  5. save the pipeline (and, for Pipeline A, the serving bundle)

Usage: python scripts/train.py [--pipelines A B] [--as-of ISO8601] [--registry DIR]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, parse_utc  # noqa: E402
from src.data.dataset import load_features, load_split  # noqa: E402
from src.evaluation.metrics import report  # noqa: E402
from src.lsh.bucket_manager import BucketTable  # noqa: E402
from src.lsh.hasher import LSHPrefilter  # noqa: E402
from src.models.model_registry import ModelRegistry  # noqa: E402
from src.pipeline import pipeline_a_rf, pipeline_b_svm, training  # noqa: E402

PIPELINES = {"A": pipeline_a_rf, "B": pipeline_b_svm}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipelines", nargs="+", default=["A", "B"], choices=list(PIPELINES))
    ap.add_argument("--as-of", default=None, help="cycle time (default: now)")
    ap.add_argument("--registry", default=None, help="model registry dir (default from config)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-bundle", action="store_true", help="do not export the serving bundle")
    args = ap.parse_args()
    cfg = load_config(args.config)
    now = parse_utc(args.as_of) if args.as_of else time.time()
    registry = ModelRegistry(Path(args.registry) if args.registry else cfg.path("registry_dir"))
    out = cfg.path("processed_dir") / "pipelines"

    df_tr, E_tr = load_split(cfg, "train")
    y_tr = df_tr["y"].to_numpy()
    X_tr = load_features(cfg, "train", E_tr)
    lsh = LSHPrefilter(cfg["lsh"])
    bids_tr = lsh.buckets(E_tr)
    table = BucketTable(cfg["buckets"]).fit(bids_tr, y_tr)
    logging.info("bucket table: %s", table.summary())

    df_va, E_va = load_split(cfg, "val")
    y_va = df_va["y"].to_numpy()
    X_va = load_features(cfg, "val", E_va)
    bids_va = lsh.buckets(E_va)

    for key in args.pipelines:
        pmod = PIPELINES[key]
        cycle = training.train_cycle(pmod, cfg, X_tr, y_tr, bids_tr, table, registry, now)
        pipe = training.assemble(pmod, cfg, table, registry)
        cal = training.calibrate(pipe, cfg, X_va, E_va, y_va, bids_va)
        scored = pipe.score(E_va, bucket_ids=bids_va, features=X_va)
        val = report(y_va, pipe.decide(scored))
        routes = {r: int((scored["routes"] == r).sum()) for r in np.unique(scored["routes"])}
        logging.info("%s calibration %s", pmod.NAME, cal)
        logging.info("%s validation: TPR %.4f FPR %.4f MCC %.4f | %s | routes %s", pmod.NAME, val["recall_tpr"],
                     val["fpr"], val["mcc"], val["false_positives"], routes)
        training.save_pipeline(pipe, out / f"{pmod.NAME}.joblib")
        training.write_json(out / f"{pmod.NAME}_train_report.json",
                            {"cycle_time": now, "bucket_table": table.summary(), "cycle": cycle,
                             "calibration": cal, "validation": val, "validation_routes": routes})
        if key == "A" and not args.no_bundle:
            training.save_pipeline(pipe, cfg.path("bundle_path"))
            logging.info("serving bundle -> %s", cfg.path("bundle_path"))


if __name__ == "__main__":
    main()
