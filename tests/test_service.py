from pathlib import Path

import joblib
import numpy as np

from defender.app import create_app
from src.features import feature_extractor as fx
from src.lsh.bucket_manager import BucketTable
from src.pipeline.inference import DetectorPipeline

FIXTURE = Path(__file__).parent / "fixtures" / "distlib_t32.exe"
LSH_CFG = {"seed": 7, "primary_mapping": "structure_focused",
           "mappings": {"structure_focused": {"rows_per_band": 3}, "content_focused": {"rows_per_band": 3}}}


class Const:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def make_app(tmp_path, p, **kw):
    table = BucketTable({"hard_region_min_malicious_samples": 1, "hard_region_min_benign_samples": 1,
                         "simple_region_min_samples": 1})
    pipe = DetectorPipeline("A_rf", fx.EXTRACTOR_VERSION, LSH_CFG, table, fallback=Const(p),
                            threshold=0.5, fallback_threshold=0.5)
    path = tmp_path / "bundle.joblib"
    joblib.dump(pipe, path)
    return create_app(path, workers=1, **kw).test_client()


def post(client, data, ctype="application/octet-stream"):
    return client.post("/", data=data, headers={"Content-Type": ctype})


def test_api_contract(tmp_path):
    benign = make_app(tmp_path, 0.1)
    r = post(benign, FIXTURE.read_bytes())
    assert r.status_code == 200 and r.get_json() == {"result": 0}
    assert benign.get("/model").get_json()["requests"] == 1
    malicious = make_app(tmp_path, 0.9)
    assert post(malicious, FIXTURE.read_bytes()).get_json() == {"result": 1}


def test_wrong_content_type_rejected(tmp_path):
    assert post(make_app(tmp_path, 0.1), b"x", "application/json").status_code == 400


def test_overrun_and_oversize_fail_safe(tmp_path):
    slow = make_app(tmp_path, 0.1, budget_seconds=0.0)
    assert post(slow, FIXTURE.read_bytes()).get_json() == {"result": 1}   # never fail open
    assert post(slow, FIXTURE.read_bytes()).get_json() == {"result": 1}   # worker was replaced
    small = make_app(tmp_path, 0.1, max_bytes=10)
    assert post(small, FIXTURE.read_bytes()).get_json() == {"result": 1}


def test_non_pe_input_scored_without_crash(tmp_path):
    assert post(make_app(tmp_path, 0.1), b"definitely not a PE").get_json() == {"result": 0}
