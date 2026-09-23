import numpy as np
import pytest

from src.drift.retrain_scheduler import Action, decide, plan_cycle
from src.evaluation.metrics import report, threshold_for_fpr
from src.features import feature_extractor as fx
from src.features.pe_parser import EMBER_DIM
from src.lsh.bucket_manager import HARD, NOVEL, SIMPLE_BENIGN, SIMPLE_MALICIOUS, BucketTable
from src.lsh.hasher import LSHPrefilter
from src.models.model_registry import ModelRegistry
from src.pipeline.inference import (ROUTE_CLASSIFIER, ROUTE_FALLBACK_HARD, ROUTE_FALLBACK_NOVEL,
                                    ROUTE_SIMPLE_BEN, ROUTE_SIMPLE_MAL, DetectorPipeline)

LSH_CFG = {"seed": 7, "primary_mapping": "structure_focused",
           "mappings": {"structure_focused": {"rows_per_band": 2}, "content_focused": {"rows_per_band": 2}}}
BUCKET_CFG = {"hard_region_min_malicious_samples": 3, "hard_region_min_benign_samples": 2,
              "simple_region_min_samples": 3}
DAY = 86400.0


def test_bucket_states():
    bids = np.array(["m"] * 4 + ["b"] * 5 + ["h"] * 6 + ["tiny"] * 2)
    y = np.array([1] * 4 + [0] * 5 + [1, 1, 1, 0, 0, 0] + [0, 0])
    t = BucketTable(BUCKET_CFG).fit(bids, y)
    assert t.state("m") == SIMPLE_MALICIOUS and t.state("b") == SIMPLE_BENIGN
    assert t.state("h") == HARD and t.state("tiny") == HARD and t.state("never-seen") == NOVEL
    assert t.trainable("h") and not t.trainable("tiny") and not t.trainable("m")
    assert t.pooled_benign == 10
    assert BucketTable.from_dict(BUCKET_CFG, t.to_dict()).state("m") == SIMPLE_MALICIOUS


@pytest.mark.parametrize("last,cleared,now,expected", [
    (None, True, 0, Action.TRAIN), (None, False, 0, Action.DEFER),
    (0.0, True, 31 * DAY, Action.RETRAIN), (0.0, False, 31 * DAY, Action.SKIP_INSUFFICIENT),
    (0.0, True, 5 * DAY, Action.SKIP_NOT_DUE), (0.0, False, 5 * DAY, Action.SKIP_NOT_DUE),
])
def test_retrain_gating(last, cleared, now, expected):
    assert decide(last, cleared, now, 30) is expected


def test_plan_cycle():
    plan = plan_cycle(["a", "b"], {"a"}, {"a": None, "b": 0.0}, 40 * DAY, 30)
    assert plan == {"a": Action.TRAIN, "b": Action.SKIP_INSUFFICIENT}


def test_registry_versions_and_rollback(tmp_path):
    r = ModelRegistry(tmp_path)
    r.save("A_rf", "b1", {"v": 1}, trained_at=0.0)
    r.save("A_rf", "b1", {"v": 2}, trained_at=31 * DAY)
    assert r.load_active("A_rf", "b1") == {"v": 2}
    assert r.last_trained_at("A_rf", "b1") == 31 * DAY
    with pytest.raises(FileExistsError):
        r.save("A_rf", "b1", {"v": 3}, trained_at=0.0)       # never overwrite
    r.rollback("A_rf", "b1")
    assert ModelRegistry(tmp_path).load_active("A_rf", "b1") == {"v": 1}
    assert len(list((tmp_path / "A_rf" / "b1").glob("*.joblib"))) == 2


def test_metrics_absolute_counts():
    rep = report(np.array([0, 0, 0, 0, 1, 1]), np.array([1, 0, 0, 0, 1, 0]))
    assert (rep["fp"], rep["fn"], rep["tp"], rep["tn"]) == (1, 1, 1, 3)
    assert rep["false_positives"].startswith("1 FP out of 4 benign")
    assert "accuracy" not in rep


def test_threshold_for_fpr():
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    t = threshold_for_fpr(s, 1)
    assert (s >= t).sum() == 1
    assert (s >= threshold_for_fpr(s, 0)).sum() == 0


def _ember(n, seed):
    rng = np.random.default_rng(seed)
    E = np.zeros((n, EMBER_DIM), dtype=np.float32)
    E[:, 616 + 62 + 255 + 256:616 + 62 + 255 + 256 + 1024] = rng.integers(-1, 2, size=(n, 1024)) * \
        (rng.random((n, 1024)) < 0.02)
    return E


def test_lsh_is_deterministic_and_query_only():
    E = _ember(20, 0)
    a, b = LSHPrefilter(LSH_CFG), LSHPrefilter(LSH_CFG)
    assert list(a.buckets(E)) == list(b.buckets(E))
    # a near-duplicate (one import flipped out of ~20) usually shares the bucket
    E2 = E.copy()
    same = 0
    for i in range(len(E)):
        nz = np.flatnonzero(E2[i])
        E2[i, nz[0]] = 0
        same += a.buckets(E[i])[0] == a.buckets(E2[i])[0]
    assert same >= len(E) // 2
    assert set(a.all_buckets(E)) == {"structure_focused", "content_focused"}


def test_feature_extractor_shape_and_version():
    X = fx.transform(_ember(3, 1))
    assert X.shape == (3, fx.N_FEATURES) and len(fx.FEATURE_NAMES) == fx.N_FEATURES
    assert "header_timestamp" in fx.FEATURE_NAMES and "resources_entropy" in fx.FEATURE_NAMES
    with pytest.raises(ValueError):
        fx.check_version("0.0.1")


class _Const:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def test_routing_and_failsafe():
    table = BucketTable(BUCKET_CFG)
    table.buckets = {}
    from src.lsh.bucket_manager import BucketInfo
    table.buckets.update({"SM": BucketInfo(0, 10, SIMPLE_MALICIOUS), "SB": BucketInfo(10, 0, SIMPLE_BENIGN),
                          "H1": BucketInfo(5, 5, HARD), "H2": BucketInfo(5, 1, HARD)})
    pipe = DetectorPipeline("A_rf", fx.EXTRACTOR_VERSION, LSH_CFG, table, classifiers={"H1": _Const(0.2)},
                            fallback=_Const(0.7), threshold=0.5, fallback_threshold=0.5)
    E = _ember(5, 2)
    bids = np.array(["SM", "SB", "H1", "H2", "NEW"], dtype=object)
    scored = pipe.score(E, bucket_ids=bids)
    assert list(scored["routes"]) == [ROUTE_SIMPLE_MAL, ROUTE_SIMPLE_BEN, ROUTE_CLASSIFIER,
                                      ROUTE_FALLBACK_HARD, ROUTE_FALLBACK_NOVEL]
    # untrained-hard and novel buckets are never auto-passed: they go to the fallback model
    assert list(pipe.decide(scored)) == [1, 0, 0, 1, 1]
