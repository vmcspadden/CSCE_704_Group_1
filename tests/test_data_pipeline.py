import numpy as np
import pandas as pd
import pytest

from src.data import labeling, sampling
from src.data.labeling import Label, consensus_label

CFG = {"min_engines_reporting": 10, "malicious_consensus_threshold": 5, "benign_consensus_threshold": 0,
       "relabel_window_days": 30}


@pytest.mark.parametrize("pos,rep,expected", [
    (0, 50, Label.BENIGN), (5, 50, Label.MALICIOUS), (30, 60, Label.MALICIOUS),
    (1, 50, Label.AMBIGUOUS), (4, 50, Label.AMBIGUOUS), (0, 5, Label.INSUFFICIENT_COVERAGE),
    (0, None, Label.BENIGN), (None, 50, Label.INSUFFICIENT_COVERAGE),
])
def test_consensus_label(pos, rep, expected):
    assert consensus_label(pos, rep, CFG) is expected


def test_apply_labels_and_reverification():
    day = 86400
    df = pd.DataFrame({
        "engines_malicious": [0, 7, 2, 0], "engines_reporting": [60, 60, 60, 3],
        "first_seen_utc": [0, 0, 0, 0], "consensus_computed_at_utc": [40 * day, 10 * day, 40 * day, 40 * day]})
    out = labeling.apply_labels(df, CFG)
    assert list(out.label) == ["BENIGN", "MALICIOUS", "AMBIGUOUS", "INSUFFICIENT_COVERAGE"]
    assert list(out.needs_reverification) == [False, True, False, False]
    kept = labeling.trainable(out)
    assert list(kept.label) == ["BENIGN"]          # ambiguous/coverage/unstable all excluded
    assert len(labeling.trainable(out, allow_unverified=True)) == 2


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        consensus_label(3, 50, {**CFG, "malicious_consensus_threshold": 0})


def _df(n_ben_by_source, n_mal):
    rows = [(f"b{s}{i}", "BENIGN", s) for s, n in n_ben_by_source.items() for i in range(n)]
    rows += [(f"m{i}", "MALICIOUS", "feedX") for i in range(n_mal)]
    df = pd.DataFrame(rows, columns=["sample_id", "label", "source"])
    df["first_seen_utc"] = np.arange(len(df), dtype=float)
    return df


def test_confounding_report_flags_but_does_not_modify():
    df = _df({"pkgmgr": 90, "vendor": 10}, 20)
    rep = sampling.confounding_report(df, 0.8)
    flagged = {(f["label"], f["source"]) for f in rep["flagged"]}
    assert flagged == {("BENIGN", "pkgmgr"), ("MALICIOUS", "feedX")}


def test_target_prevalence_downsamples_benign_only_stratified():
    df = _df({"a": 600, "b": 400}, 50)
    out = sampling.target_prevalence(df, 10.0, seed=0)
    assert (out.label == "MALICIOUS").sum() == 50
    ben = out[out.label == "BENIGN"]
    assert len(ben) == 500
    assert (ben.source == "a").sum() == 300 and (ben.source == "b").sum() == 200


def test_target_prevalence_never_oversamples():
    df = _df({"a": 10}, 50)
    out = sampling.target_prevalence(df, 3.0, seed=0)
    assert len(out) == len(df)


def test_budget_subsample_caps():
    df = _df({"a": 100}, 50)
    out = sampling.budget_subsample(df, {"BENIGN": 10}, seed=0)
    assert (out.label == "BENIGN").sum() == 10 and (out.label == "MALICIOUS").sum() == 50
