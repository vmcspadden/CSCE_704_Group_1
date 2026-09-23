"""Raw manifest -> labeled, temporally split, sampled manifests.

Order of operations (each step only sees what the previous one produced):
  1. load the raw manifest (SOREL-20M meta.db: sha256, first-seen time,
     engine detection counts) - IDs and metadata only, no features
  2. §3.1 consensus labeling; drop AMBIGUOUS / INSUFFICIENT_COVERAGE /
     not-yet-stable samples
  3. §3.2 temporal split (first step that partitions data)
  4. §3.3 confounding report (per split) + budget caps + realistic
     prevalence for val/test
  5. write data/splits/{train,val,test}.csv and data/splits/dataset_report.json

Features are fetched afterwards, per split, by scripts/fetch_features.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.data import labeling, sampling  # noqa: E402
from src.data.sorel_loader import META_DB_URL  # noqa: E402
from src.data.temporal_split import SplitBoundaries, temporal_split  # noqa: E402

MISSING_URL = "https://raw.githubusercontent.com/sophos/SOREL-20M/master/shas_missing_ember_features.json"
SOREL_CONSENSUS_TIME = 1607472000.0  # 2020-12-09: SOREL counts were frozen at publication


def load_sorel_manifest(meta_db: Path) -> pd.DataFrame:
    con = sqlite3.connect(meta_db)
    df = pd.read_sql_query(
        "select sha256 as sample_id, rl_fs_t as first_seen_utc, "
        "rl_ls_const_positives as engines_malicious from meta", con)
    con.close()
    df["engines_reporting"] = np.nan  # not published by SOREL
    df["consensus_computed_at_utc"] = SOREL_CONSENSUS_TIME
    df["source"] = "sorel-20m"
    missing = set(requests.get(MISSING_URL, timeout=60).json())
    return df[~df["sample_id"].isin(missing)].reset_index(drop=True)


def ensure_meta_db(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with requests.get(META_DB_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    tmp.rename(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    meta_db = cfg.path("raw_dir") / "sorel" / "meta.db"
    ensure_meta_db(meta_db)
    raw = load_sorel_manifest(meta_db)
    logging.info("raw manifest: %d samples", len(raw))

    labeled = labeling.apply_labels(raw, cfg["labeling"])
    report = {"labeling": labeling.label_summary(labeled)}
    labeled = labeling.trainable(labeled)

    splits = temporal_split(labeled, SplitBoundaries.from_config(cfg["temporal_split"]))
    s = cfg["sampling"]
    out_dir = cfg.path("splits_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report["splits"] = {}
    for i, (name, df) in enumerate(splits.items()):
        entry = {"after_split": df["label"].value_counts().to_dict(),
                 "confounding": sampling.confounding_report(df, s["max_single_source_fraction"])}
        df = sampling.budget_subsample(df, s["budget"].get(name, {}), seed=s["random_seed"] + i)
        if name != "train":
            df = sampling.target_prevalence(df, s["eval_benign_to_malicious_ratio"], seed=s["random_seed"] + i)
        entry["final"] = df["label"].value_counts().to_dict()
        entry["first_seen_range"] = [float(df["first_seen_utc"].min()), float(df["first_seen_utc"].max())]
        report["splits"][name] = entry
        cols = ["sample_id", "first_seen_utc", "label", "y", "source", "engines_malicious"]
        df[cols].to_csv(out_dir / f"{name}.csv", index=False)
        logging.info("%s: %s", name, entry["final"])
    with open(out_dir / "dataset_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
