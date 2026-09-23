"""HTTP service exposing Pipeline A (LSH pre-filter -> per-bucket RF).

API (same contract as the MLSEC defender challenge):
  POST /        body = raw PE bytes, Content-Type: application/octet-stream
                -> {"result": 0} benign | {"result": 1} malicious
  GET  /model   -> model / pipeline metadata

Each sample is scored in a worker process with a hard deadline. The parent
loads the bundle once and forks workers from it (model memory is shared
copy-on-write). If a worker exceeds the per-request budget, or crashes on a
malformed file, it is killed and replaced, and the request gets the
fail-safe verdict 1 (§10.2: never fail open) - a timeout would otherwise be
counted as an evasion.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("defender")
_CTX = mp.get_context("fork")
_PIPE = None  # loaded in the parent before workers fork


def _worker_loop(conn) -> None:
    from src.features.pe_parser import ember_vector

    while True:
        try:
            bytez = conn.recv_bytes()
        except (EOFError, OSError):
            return
        try:
            vec, _ = ember_vector(bytez)
            result = int(_PIPE.predict(vec[None, :])[0])
        except Exception:  # noqa: BLE001 - unscoreable input -> fail-safe malicious
            result = 1
        conn.send(result)


class Worker:
    def __init__(self):
        self.conn, child = _CTX.Pipe()
        self.proc = _CTX.Process(target=_worker_loop, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.join(1)
        finally:
            self.conn.close()


class WorkerPool:
    def __init__(self, n: int):
        self.idle: queue.Queue[Worker] = queue.Queue()
        for _ in range(n):
            self.idle.put(Worker())

    def classify(self, bytez: bytes, deadline: float) -> tuple[int, str]:
        try:
            w = self.idle.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            return 1, "queue_timeout"
        try:
            w.conn.send_bytes(bytez)
            if w.conn.poll(max(0.0, deadline - time.monotonic())):
                result = w.conn.recv()
                self.idle.put(w)
                return int(result), "ok"
            status = "timeout"
        except (EOFError, OSError, BrokenPipeError):
            status = "worker_died"
        w.kill()
        self.idle.put(Worker())
        return 1, status


def create_app(bundle_path: str | os.PathLike, workers: int = 2, budget_seconds: float = 4.5,
               max_bytes: int = 16 * 1024 * 1024) -> Flask:
    global _PIPE
    _PIPE = joblib.load(bundle_path)
    # warm up LIEF / sklearn code paths in the parent so forked workers inherit them
    from src.features.pe_parser import ember_vector
    _PIPE.predict(ember_vector(b"MZ" + b"\0" * 1024)[0][None, :])
    pool = WorkerPool(workers)
    info = {
        "name": _PIPE.name,
        "extractor_version": _PIPE.extractor_version,
        "threshold": _PIPE.threshold,
        "fallback_threshold": _PIPE.fallback_threshold,
        "n_buckets": len(_PIPE.buckets.buckets),
        "n_bucket_classifiers": len(_PIPE.classifiers),
        "workers": workers,
        "request_budget_seconds": budget_seconds,
    }

    app = Flask(__name__)
    lock = threading.Lock()
    stats = {"requests": 0, "timeouts": 0}

    @app.route("/", methods=["POST"])
    def post():
        start = time.monotonic()
        if request.headers.get("Content-Type") != "application/octet-stream":
            resp = jsonify({"error": "expecting application/octet-stream"})
            resp.status_code = 400
            return resp
        bytez = request.get_data(cache=False)
        if len(bytez) > max_bytes:
            result, status = 1, "too_large"
        else:
            result, status = pool.classify(bytez, start + budget_seconds)
        with lock:
            stats["requests"] += 1
            stats["timeouts"] += status != "ok"
        if status != "ok":
            log.warning("fail-safe verdict (%s) after %.2fs", status, time.monotonic() - start)
        return jsonify({"result": result})

    @app.route("/model", methods=["GET"])
    def model_info():
        return jsonify({**info, **stats})

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = os.environ
    bundle = env.get("DF_BUNDLE_PATH", str(Path(__file__).resolve().parent / "model" / "bundle.joblib"))
    app = create_app(bundle, workers=int(env.get("DF_WORKERS", "2")),
                     budget_seconds=float(env.get("DF_REQUEST_BUDGET", "4.5")),
                     max_bytes=int(env.get("DF_MAX_BYTES", str(16 * 1024 * 1024))))
    from waitress import serve

    port = int(env.get("DF_PORT", "8080"))
    log.info("serving on :%d", port)
    serve(app, host="0.0.0.0", port=port, threads=8, max_request_body_size=64 * 1024 * 1024)


if __name__ == "__main__":
    main()
