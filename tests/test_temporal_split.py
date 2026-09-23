"""Correctness-critical tests for the §3.2 temporal split."""
import pandas as pd
import pytest

from src.data.temporal_split import (SplitBoundaries, TemporalSplitError, check_split_integrity,
                                     dedupe_first_seen, temporal_split)

B = SplitBoundaries(train_end=100.0, val_end=200.0, test_start=200.0)


def manifest(rows):
    return pd.DataFrame(rows, columns=["sample_id", "first_seen_utc"])


def basic():
    return manifest([("a", 10), ("b", 100), ("c", 150), ("d", 199.9), ("e", 200), ("f", 500)])


def test_assigns_by_first_seen():
    s = temporal_split(basic(), B)
    assert list(s["train"].sample_id) == ["a", "b"]          # <= train_end
    assert list(s["val"].sample_id) == ["c", "d"]            # (train_end, val_end)
    assert list(s["test"].sample_id) == ["e", "f"]           # >= test_start


def test_splits_strictly_time_ordered():
    s = temporal_split(basic(), B)
    assert s["train"].first_seen_utc.max() < s["val"].first_seen_utc.min()
    assert s["val"].first_seen_utc.max() < s["test"].first_seen_utc.min()


@pytest.mark.parametrize("rows,empty", [
    ([("a", 150), ("b", 300)], "train"),
    ([("a", 10), ("b", 300)], "val"),
    ([("a", 10), ("b", 150)], "test"),
])
def test_empty_split_raises(rows, empty):
    with pytest.raises(TemporalSplitError, match=empty):
        temporal_split(manifest(rows), B)


def test_duplicate_id_across_windows_lands_in_one_split_only():
    # "x" appears once in the train window and once in the test window.
    df = manifest([("a", 10), ("x", 50), ("c", 150), ("x", 400), ("f", 500)])
    s = temporal_split(df, B)
    all_ids = pd.concat([s[k].sample_id for k in s])
    assert all_ids.is_unique
    assert "x" in set(s["train"].sample_id)     # earliest sighting defines first_seen
    assert "x" not in set(s["test"].sample_id)


def test_duplicate_id_val_and_test():
    df = manifest([("a", 10), ("y", 199), ("y", 250), ("c", 150), ("f", 500)])
    s = temporal_split(df, B)
    assert "y" in set(s["val"].sample_id) and "y" not in set(s["test"].sample_id)


def test_dedupe_keeps_earliest():
    d = dedupe_first_seen(manifest([("x", 400), ("x", 50), ("x", 150)]))
    assert len(d) == 1 and d.first_seen_utc.iloc[0] == 50


def test_gap_between_val_and_test_is_dropped():
    b = SplitBoundaries(train_end=100.0, val_end=200.0, test_start=300.0)
    s = temporal_split(manifest([("a", 10), ("c", 150), ("g", 250), ("f", 500)]), b)
    assert "g" not in set(pd.concat([s[k].sample_id for k in s]))


def test_boundary_validation():
    with pytest.raises(TemporalSplitError):
        temporal_split(basic(), SplitBoundaries(200.0, 100.0, 300.0))
    with pytest.raises(TemporalSplitError):
        temporal_split(basic(), SplitBoundaries(100.0, 200.0, 150.0))


def test_missing_timestamp_raises():
    df = manifest([("a", 10), ("b", None), ("c", 150), ("f", 500)])
    with pytest.raises(TemporalSplitError):
        temporal_split(df, B)


def test_integrity_check_detects_overlap_and_shared_ids():
    ok = {"train": manifest([("a", 1)]), "val": manifest([("b", 2)]), "test": manifest([("c", 3)])}
    check_split_integrity(ok)
    with pytest.raises(TemporalSplitError, match="overlap"):
        check_split_integrity({**ok, "val": manifest([("b", 0.5)])})
    with pytest.raises(TemporalSplitError, match="in both"):
        check_split_integrity({**ok, "test": manifest([("a", 3)])})
    with pytest.raises(TemporalSplitError, match="duplicate"):
        check_split_integrity({**ok, "test": manifest([("c", 3), ("c", 4)])})


def test_from_config_parses_iso_timestamps():
    b = SplitBoundaries.from_config({"train_end": "1970-01-01T00:01:40Z", "val_end": "1970-01-01T00:03:20Z"})
    assert (b.train_end, b.val_end, b.test_start) == (100.0, 200.0, 200.0)
