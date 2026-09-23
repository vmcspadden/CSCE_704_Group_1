"""Fetch EMBER v2 vectors for the split manifests from SOREL-20M's LMDB.

Runs after scripts/build_dataset.py. For each split CSV it writes
data/processed/ember/<split>.npy (float32, one row per manifest row, raw
EMBER scale) and <split>.found.npy (bool; False where SOREL has no vector).
Progress is checkpointed, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.data.sorel_loader import EMBER_DIM, LMDBHttpReader, decode_ember_value  # noqa: E402


def fetch_split(reader: LMDBHttpReader, ids: list[str], out: Path, threads: int, chunk: int) -> None:
    n = len(ids)
    x_path, done_path, found_path = out.with_suffix(".npy"), out.with_suffix(".done.npy"), \
        out.with_suffix(".found.npy")
    mode = "r+" if x_path.exists() else "w+"
    X = np.lib.format.open_memmap(x_path, mode=mode, dtype=np.float32, shape=(n, EMBER_DIM))
    done = np.load(done_path) if done_path.exists() else np.zeros(n, dtype=bool)
    found = np.load(found_path) if found_path.exists() else np.zeros(n, dtype=bool)
    todo = np.flatnonzero(~done)
    print(f"{out.name}: {n} samples, {len(todo)} to fetch", flush=True)

    def one(i: int) -> tuple[int, np.ndarray | None]:
        raw = reader.get(ids[i])
        return i, (None if raw is None else decode_ember_value(raw))

    t0 = time.time()
    with cf.ThreadPoolExecutor(threads) as pool:
        for c in range(0, len(todo), chunk):
            for i, vec in pool.map(one, todo[c:c + chunk]):
                if vec is not None:
                    X[i] = vec
                    found[i] = True
                done[i] = True
            X.flush()
            np.save(done_path, done)
            np.save(found_path, found)
            k = min(c + chunk, len(todo))
            rate = k / max(1e-9, time.time() - t0)
            print(f"  {k}/{len(todo)}  {rate:.0f}/s  eta {(len(todo) - k) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)
    done_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=5000)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = cfg.path("processed_dir") / "ember"
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = LMDBHttpReader()
    for split in args.splits:
        ids = pd.read_csv(cfg.path("splits_dir") / f"{split}.csv", usecols=["sample_id"])["sample_id"].tolist()
        fetch_split(reader, ids, out_dir / split, args.threads, args.chunk)


if __name__ == "__main__":
    main()
