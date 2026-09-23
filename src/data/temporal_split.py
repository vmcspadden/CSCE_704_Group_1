"""§3.2 strict temporal train/validation/test split.

This is the first pipeline step: it runs on the raw sample manifest before
feature extraction, LSH fitting or model training touch any sample.

  train: first_seen_utc <= train_end
  val:   train_end < first_seen_utc < val_end
  test:  first_seen_utc >= test_start (test_start >= val_end; anything in the
         optional gap [val_end, test_start) is dropped)

No k-fold cross-validation is used anywhere in this project.

Guarantees (tested in tests/test_temporal_split.py):
  * no split is empty (raises TemporalSplitError),
  * splits are strictly time-ordered relative to each other,
  * no sample_id appears in more than one split, even when duplicate manifest
    rows for one ID carry timestamps in different windows. Such an ID is
    resolved to its earliest first_seen_utc, which is the definition of
    "first seen".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import parse_utc


class TemporalSplitError(ValueError):
    pass


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: float
    val_end: float
    test_start: float

    @classmethod
    def from_config(cls, cfg) -> "SplitBoundaries":
        train_end = parse_utc(cfg["train_end"])
        val_end = parse_utc(cfg["val_end"])
        test_start = parse_utc(cfg.get("test_start") or cfg["val_end"])
        return cls(train_end, val_end, test_start)

    def validate(self) -> None:
        if not self.train_end < self.val_end:
            raise TemporalSplitError("train_end must be strictly before val_end")
        if self.test_start < self.val_end:
            raise TemporalSplitError("test_start must not be before val_end")


def dedupe_first_seen(manifest: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate sample_id rows to the earliest first_seen_utc."""
    if "sample_id" not in manifest or "first_seen_utc" not in manifest:
        raise TemporalSplitError("manifest needs sample_id and first_seen_utc columns")
    df = manifest.copy()
    df["first_seen_utc"] = pd.to_numeric(df["first_seen_utc"], errors="raise")
    if df["first_seen_utc"].isna().any():
        raise TemporalSplitError("first_seen_utc has missing values")
    df = df.sort_values(["first_seen_utc", "sample_id"], kind="mergesort")
    return df.drop_duplicates("sample_id", keep="first")


def temporal_split(manifest: pd.DataFrame, bounds: SplitBoundaries) -> dict[str, pd.DataFrame]:
    bounds.validate()
    df = dedupe_first_seen(manifest)
    t = df["first_seen_utc"]
    splits = {
        "train": df[t <= bounds.train_end],
        "val": df[(t > bounds.train_end) & (t < bounds.val_end)],
        "test": df[t >= bounds.test_start],
    }
    for name, part in splits.items():
        if part.empty:
            raise TemporalSplitError(f"{name} split is empty; check the split boundaries")
    check_split_integrity(splits)
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def check_split_integrity(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if splits overlap in time or share any sample_id."""
    order = ["train", "val", "test"]
    for a, b in zip(order, order[1:]):
        if splits[a]["first_seen_utc"].max() >= splits[b]["first_seen_utc"].min():
            raise TemporalSplitError(f"{a} and {b} overlap in time")
    for name in order:
        if splits[name]["sample_id"].duplicated().any():
            raise TemporalSplitError(f"duplicate sample_id inside {name}")
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            shared = splits[a]["sample_id"][splits[a]["sample_id"].isin(splits[b]["sample_id"])]
            if not shared.empty:
                raise TemporalSplitError(f"sample_id {shared.iloc[0]} in both {a} and {b}")
