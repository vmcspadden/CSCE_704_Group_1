"""Black-box test of the running HTTP service (like the MLSEC defender test).

  python scripts/test_service.py -b data/raw/benign_pypi -m data/raw/sorel_binaries

Posts every file in the benign (-b) and malicious (-m) folders to the
service and reports FPR, TPR, errors and latency (max / p99). A request that
takes longer than --timeout counts as a miss, as the grader would count it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests


def run(folder: Path, url: str, timeout: float, limit: int) -> tuple[list[int | None], list[float]]:
    results, latencies = [], []
    for f in sorted(p for p in folder.iterdir() if p.is_file())[:limit]:
        t0 = time.monotonic()
        try:
            r = requests.post(url, data=f.read_bytes(), headers={"Content-Type": "application/octet-stream"},
                              timeout=timeout)
            results.append(int(r.json()["result"]))
        except (requests.RequestException, ValueError, KeyError):
            results.append(None)
        latencies.append(time.monotonic() - t0)
    return results, latencies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-b", type=Path, required=True, help="folder of benign files")
    ap.add_argument("-m", type=Path, required=True, help="folder of malicious files")
    ap.add_argument("--url", default="http://127.0.0.1:8080/")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    ben, lb = run(args.b, args.url, args.timeout, args.limit)
    mal, lm = run(args.m, args.url, args.timeout, args.limit)
    lat = np.array(lb + lm)
    summary = {
        "benign": {"n": len(ben), "false_positives": sum(r == 1 for r in ben), "errors": ben.count(None),
                   "fpr": sum(r == 1 for r in ben) / max(1, len(ben))},
        "malicious": {"n": len(mal), "detected": sum(r == 1 for r in mal), "errors": mal.count(None),
                      "tpr": sum(r == 1 for r in mal) / max(1, len(mal))},
        "latency_seconds": {"mean": float(lat.mean()), "p99": float(np.percentile(lat, 99)),
                            "max": float(lat.max())},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
