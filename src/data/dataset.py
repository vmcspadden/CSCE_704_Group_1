"""Load split manifests with their EMBER vectors and versioned model features."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.features import feature_extractor as fx


def load_split(cfg, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """(manifest rows with a SOREL vector, EMBER vectors [n, 2381])."""
    df = pd.read_csv(cfg.path("splits_dir") / f"{split}.csv")
    ember_dir = cfg.path("processed_dir") / "ember"
    found = np.load(ember_dir / f"{split}.found.npy")
    if (ember_dir / f"{split}.done.npy").exists():
        raise RuntimeError(f"feature download for {split} is incomplete; rerun scripts/fetch_features.py")
    E = np.load(ember_dir / f"{split}.npy", mmap_mode="r")
    idx = np.flatnonzero(found)
    return df.iloc[idx].reset_index(drop=True), np.asarray(E[idx])


def load_features(cfg, split: str, E: np.ndarray) -> np.ndarray:
    """Model features for a split, cached per extractor version (§4.2: a new
    version means a fresh extraction, never an in-place update)."""
    d = cfg.path("processed_dir") / "features" / f"v{fx.EXTRACTOR_VERSION}"
    path, meta = d / f"{split}.npy", d / f"{split}.json"
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text())
        fx.check_version(info["extractor_version"])
        if info["n"] == len(E):
            return np.load(path)
    d.mkdir(parents=True, exist_ok=True)
    X = np.vstack([fx.transform(E[i:i + 20000]) for i in range(0, len(E), 20000)])
    np.save(path, X)
    meta.write_text(json.dumps({"extractor_version": fx.EXTRACTOR_VERSION, "n": len(E)}))
    return X
