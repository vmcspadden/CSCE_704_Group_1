"""Check that src/features/pe_parser.py reproduces SOREL's EMBER v2 vectors.

Downloads N malicious binaries from SOREL-20M's (disarmed) binary store for
test-split samples, extracts EMBER v2 vectors from the raw bytes, and
compares them block by block with the vectors SOREL computed with LIEF 0.9.

SOREL disarmed its binaries by zeroing FileHeader.Machine and
OptionalHeader.Subsystem, so the machine/subsystem hash bins (and the few
byte-histogram counts those bytes touch) are expected to differ and are
excluded from the header comparison. The binaries are only parsed, never
modified or executed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.data.sorel_loader import SOREL_BUCKET, LMDBHttpReader, decode_ember_value  # noqa: E402
from src.features.pe_parser import BLOCKS, OFFSETS, ember_vector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_config()
    out = cfg.path("raw_dir") / "sorel_binaries"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.path("splits_dir") / f"{args.split}.csv")
    shas = df[df.y == 1].sample(args.n, random_state=0).sample_id.tolist()
    reader = LMDBHttpReader()

    def fetch(sha: str):
        path = out / sha
        if not path.exists():
            r = requests.get(f"{SOREL_BUCKET}/binaries/{sha}", timeout=120)
            if r.status_code != 200:
                return None
            path.write_bytes(zlib.decompress(r.content))
        ref = reader.get(sha)
        return None if ref is None else (path.read_bytes(), decode_ember_value(ref))

    with cf.ThreadPoolExecutor(32) as pool:
        pairs = [p for p in pool.map(fetch, shas) if p is not None]

    mismatched = {name: 0 for name, _ in BLOCKS}
    for raw, ref in pairs:
        mine, _ = ember_vector(raw)
        for name, (a, b) in OFFSETS.items():
            diff = np.abs(mine[a:b] - ref[a:b]) > 1e-3 * (np.abs(ref[a:b]) + 1e-3)
            if name == "header":
                diff[1:11] = False      # machine hash bins (zeroed by disarming)
                diff[21:31] = False     # subsystem hash bins (zeroed by disarming)
            if name == "histogram" and diff.sum() <= 4:
                diff[:] = False         # the disarmed header bytes shift <= 4 byte counts
            mismatched[name] += bool(diff.any())
    result = {"n_samples": len(pairs),
              "fraction_of_samples_with_any_mismatch": {k: round(v / max(1, len(pairs)), 4)
                                                         for k, v in mismatched.items()}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
