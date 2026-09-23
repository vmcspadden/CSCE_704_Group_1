"""§8 fixed-cadence retraining with per-bucket gating.

Active drift detectors (ADWIN, DDM, EDDM) are deliberately NOT used: with
AV labels that take 19-30 days to stabilize, they discard "stale" data
while its labels are still arriving. A fixed 30-day cadence avoids that.

Per hard-region bucket, at each scheduled cycle (§8.2):

  never trained, threshold cleared     -> TRAIN
  never trained, threshold not cleared -> DEFER (served by the §10.2 fallback)
  trained, cadence due, cleared        -> RETRAIN
  trained, cadence due, not cleared    -> SKIP_INSUFFICIENT (keep last model)
  trained, cadence not due             -> SKIP_NOT_DUE

Simple regions are never scheduled; they update by lookup as labeled
samples land in them.
"""
from __future__ import annotations

from enum import Enum

SECONDS_PER_DAY = 86400.0


class Action(str, Enum):
    TRAIN = "train"
    DEFER = "defer"
    RETRAIN = "retrain"
    SKIP_INSUFFICIENT = "skip_insufficient"
    SKIP_NOT_DUE = "skip_not_due"


def decide(last_trained_at: float | None, threshold_cleared: bool, now: float, cadence_days: float) -> Action:
    if last_trained_at is None:
        return Action.TRAIN if threshold_cleared else Action.DEFER
    if now - last_trained_at < cadence_days * SECONDS_PER_DAY:
        return Action.SKIP_NOT_DUE
    return Action.RETRAIN if threshold_cleared else Action.SKIP_INSUFFICIENT


def plan_cycle(hard_buckets: list[str], trainable: set[str], last_trained: dict[str, float | None],
               now: float, cadence_days: float) -> dict[str, Action]:
    return {b: decide(last_trained.get(b), b in trainable, now, cadence_days) for b in hard_buckets}


def should_train(action: Action) -> bool:
    return action in (Action.TRAIN, Action.RETRAIN)
