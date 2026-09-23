"""§8.3 versioned per-bucket model storage with rollback.

Every trained model is written to a new timestamped file and never
overwritten:

  <registry>/<pipeline>/<bucket>/<YYYYmmddTHHMMSSZ>.joblib

``manifest.json`` records, per (pipeline, bucket), the active version, the
full history, and when the bucket was last trained (read by the retrain
scheduler).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib


def _stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def _entry(self, pipeline: str, bucket: str) -> dict:
        return self.manifest.setdefault(pipeline, {}).setdefault(bucket, {"active": None, "history": []})

    def save(self, pipeline: str, bucket: str, model, trained_at: float, metadata: dict | None = None) -> Path:
        path = self.root / pipeline / bucket / f"{_stamp(trained_at)}.joblib"
        if path.exists():
            raise FileExistsError(f"{path} already exists; models are never overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)
        entry = self._entry(pipeline, bucket)
        rel = str(path.relative_to(self.root))
        entry["history"].append({"path": rel, "trained_at": trained_at, **(metadata or {})})
        entry["active"] = rel
        entry["last_trained_at"] = trained_at
        self._write()
        return path

    def active_path(self, pipeline: str, bucket: str) -> Path | None:
        entry = self.manifest.get(pipeline, {}).get(bucket)
        return None if not entry or not entry["active"] else self.root / entry["active"]

    def load_active(self, pipeline: str, bucket: str):
        p = self.active_path(pipeline, bucket)
        return None if p is None else joblib.load(p)

    def last_trained_at(self, pipeline: str, bucket: str) -> float | None:
        return self.manifest.get(pipeline, {}).get(bucket, {}).get("last_trained_at")

    def buckets(self, pipeline: str) -> list[str]:
        return sorted(b for b, e in self.manifest.get(pipeline, {}).items() if e["active"])

    def rollback(self, pipeline: str, bucket: str) -> Path:
        """Re-activate the version trained before the currently active one."""
        entry = self.manifest[pipeline][bucket]
        paths = [h["path"] for h in entry["history"]]
        i = paths.index(entry["active"])
        if i == 0:
            raise ValueError(f"no earlier model for {pipeline}/{bucket}")
        entry["active"] = paths[i - 1]
        self._write()
        return self.root / entry["active"]

    def _write(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1, sort_keys=True))
        tmp.replace(self.manifest_path)
