# Static PE malware detector

A static, ML-based detector for Windows PE files, packaged as a self-contained
Docker image with an HTTP API. It implements the design spec
(`pe-malware-detector-DESIGN.md`): a MinHash-LSH pre-filter routes each file to a
*simple* region (decided by lookup) or a *hard* region (decided by a per-bucket
Random Forest). A calibrated Linear SVM pipeline serves as the baseline.

| Requirement | Target | Status (see [Results](#results)) |
|---|---|---|
| Delivery | Docker image, HTTP | `POST /` on port 8080 (MLSEC defender API) |
| False positive rate | ≤ 1% | ✅ 0.50% validation, 0.505% test, 0 / 536 on modern benign binaries |
| True positive rate | ≥ 95% | ✅ 95.5% validation; ⚠️ 92.9% on the later test period (drift, see below) |
| Memory | ≤ 1 GB | ✅ ~315 MiB idle, ~520 MiB under load |
| Response time | ≤ 5 s | ✅ mean 0.07 s, p99 0.5 s; hard 4.5 s budget with fail-safe verdict |

## Quick start

```bash
docker build -t pe-detector .
docker run --rm -p 8080:8080 --memory=1g --cpus=1 pe-detector

curl -XPOST --data-binary @some.exe -H "Content-Type: application/octet-stream" http://127.0.0.1:8080/
# {"result": 0}   0 = benign, 1 = malicious
curl http://127.0.0.1:8080/model        # pipeline metadata + request counters
```

The trained model bundle (`defender/model/bundle.joblib`) is committed, so the image
builds without retraining. Test a running container on your own sample folders with:

```bash
python scripts/test_service.py -b <benign_dir> -m <malicious_dir>
```

Environment variables (all optional): `DF_WORKERS` (scoring processes, default 2),
`DF_REQUEST_BUDGET` (seconds, default 4.5), `DF_PORT`, `DF_MAX_BYTES`, `DF_BUNDLE_PATH`.

## Architecture

```
PE bytes ─► pe_parser (EMBER v2 raw features on LIEF) ─► 2381-d EMBER vector
                                                            │
                   ┌────────────────────────────────────────┤
                   ▼                                        ▼
   feature_extractor (versioned, 2394 named features)   LSH pre-filter (MinHash)
                   │                                        │ structure_focused mapping → bucket id
                   │                                        ▼
                   │       simple_malicious ─► 1        simple_benign ─► 0
                   │       hard + classifier  ─► bucket RF  (score ≥ threshold)
                   └──────► hard w/o classifier / novel bucket ─► secondary-review RF (fail-safe route)
```

Serving: the HTTP parent loads the bundle once and forks scoring workers, which
share model memory copy-on-write. A worker that overruns the time budget or crashes
on a malformed file is killed and replaced, and that request returns `1`. The spec's
§10.2 fail-safe says to never fail open, and a timeout would otherwise count as an
evasion.

## How the design spec maps to code

| Spec | Where | Notes |
|---|---|---|
| §3.1 consensus labeling | `src/data/labeling.py` | BENIGN ≤ 1 engine, MALICIOUS ≥ 5; ambiguous / low-coverage / unstable samples excluded |
| §3.2 temporal split (first step) | `src/data/temporal_split.py`, `tests/test_temporal_split.py` | non-empty, strictly time-ordered, no shared IDs (duplicates resolved to earliest sighting); no k-fold anywhere |
| §3.3 confounding / prevalence | `src/data/sampling.py` | report-only confounding check; benign-only stratified downsampling to 3:1 for val/test |
| §4 features + versioning | `src/features/pe_parser.py`, `feature_extractor.py` | `EXTRACTOR_VERSION` checked on every load; cached features stored per version |
| §5 LSH, multiple mappings | `src/lsh/hasher.py` | `structure_focused` (primary, §5.2 option a) + `content_focused` (auxiliary) |
| §6 bucket states | `src/lsh/bucket_manager.py` | two states; built from training data only |
| §7 RF / SVM, pooled benign, per-bucket imbalance | `src/models/`, `src/pipeline/pipeline_{a_rf,b_svm}.py` | `balanced_subsample` / `balanced`; SVM calibrated for `predict_proba` |
| §8 30-day cadence + gating, versioned registry | `src/drift/retrain_scheduler.py`, `src/models/model_registry.py` | timestamped model files, manifest, rollback |
| §9 SHAP audit | `src/explainability/shap_audit.py` | after every RF retrain; per-cycle report in `data/processed/shap_reports/` |
| §10 routing + fail-safe | `src/pipeline/inference.py` | untrained-hard and novel buckets → secondary-review model |
| §11 metrics, full pipeline, test discipline | `src/evaluation/metrics.py`, `scripts/evaluate.py` | absolute FP/FN counts, MCC; the test split is guarded by a one-time marker |

### Decisions made during implementation

These are the §13 open items plus places where the spec gave a default that was
tuned on the **validation** split (never test):

1. **Canonical bucket id (§5.2):** option (a). `structure_focused` alone assigns
   buckets. The `content_focused` bucket is computed and kept as an auxiliary signal.
2. **Query-time resolution (§13.2):** the LSH mapping's only parameters are seeded
   permutations, and bucket contents come from the training split only, so val/test
   lookups never refit anything.
3. **Raw-data loader (§13.3):** training uses SOREL-20M's pre-extracted EMBER v2
   vectors, read from its LMDB over HTTP (`src/data/sorel_loader.py`). At inference,
   `pe_parser.py` computes the same EMBER v2 vector from raw bytes, and both paths go
   through the same `feature_extractor.transform`. Fidelity against SOREL's stored
   vectors is checked by `scripts/check_extractor_fidelity.py` (see DATA_SOURCING.md).
   `pefile` was not used, because the training vectors come from LIEF.
4. **Authenticode (§13.4):** presence only (`signature_present`, certificate-table
   size). No chain validation.
5. **Fallback for hard buckets with no classifier, and novel buckets (§10.2/10.3):**
   routed to a "secondary review" Random Forest trained on the whole training split.
   They are never auto-passed as benign. That model's threshold can be biased further
   toward malicious with `calibration.fallback_failsafe_factor` (default 1.0, because
   the 1% FPR budget binds).
6. **Tiny single-class buckets:** a pure bucket needs ≥ 30 training samples
   (`simple_region_min_samples`) before a lookup alone is trusted. Smaller ones are
   hard regions without a classifier. This is not a third state.
7. **`hard_region_min_malicious_samples` = 2000 (spec default 50).** On validation,
   bucket RFs that had seen only a few hundred malicious samples missed much more new
   malware than the fallback (82.7% vs 94.3% TPR on the same samples). Buckets with
   ≥ 2000 malicious samples match or beat it. The 30-day scheduler lowers nothing
   automatically: as buckets accumulate samples they become trainable.
8. **LSH granularity:** `rows_per_band = 3` (~15k buckets). With 1–2, a few hundred
   very mixed buckets form and almost nothing is simple. With 3, ~11% of validation
   traffic is resolved by lookup at 99.95% (malicious) / 99.7% (benign) purity.
9. **RF size:** 100 trees (spec starting point 500), fully grown. This keeps the
   serving image well under 1 GB of RAM. Pooled benign per bucket is capped at 40k
   (180k made no measurable difference on validation).
10. **Threshold:** one threshold for all model-routed samples, chosen on validation
    so the *full pipeline* FPR, simple-region false positives included, is 0.5%.
11. **Features beyond the §4.1 table:** the EMBER byte histogram and byte/entropy
    histogram are kept as a `bytes` category. `resources_entry_count` and
    `header_entropy` (SHAP watch-list) cannot be derived from the EMBER v2 schema;
    the audit reports them as absent.
12. **Benign consensus threshold = 1** (spec: "typically 0"). See DATA_SOURCING.md.
13. **SVM solver:** the baseline is a hinge-loss linear SVM trained with
    `SGDClassifier`, not liblinear's `LinearSVC`. The model family is the same, but
    liblinear needed ~460 s for a single 42k-sample fit on these features. Its
    fallback uses 150k training samples (RF: 300k), because SGD + calibration work
    in float64 and 300k × 2394 exceeded this machine's 14 GB.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/build_dataset.py        # meta.db → labeled, split, sampled manifests (~5 min)
python scripts/fetch_features.py       # EMBER vectors for 440k samples from SOREL's LMDB (~45 min)
python scripts/train.py                # one retrain cycle for pipelines A and B; exports defender/model/bundle.joblib
python scripts/evaluate.py             # the single test-split A/B comparison
python scripts/check_extractor_fidelity.py
python scripts/collect_benign_pe.py    # optional out-of-distribution benign corpus
python -m pytest tests
```

Later cycles: rerun `scripts/train.py` every 30 days after refreshing the manifests.
The registry (`data/processed/registry/manifest.json`) records when each bucket was
last trained. Buckets that are not due, or not above threshold, keep their current
model. `ModelRegistry.rollback()` restores a previous model version.

## Results

All numbers are for the **full pipeline** (simple-region lookups + bucket RFs +
fallback). Both pipelines share one LSH mapping and bucket table. Data: SOREL-20M,
temporal split by first-seen time.

| Split | Period | Benign : malicious |
|---|---|---|
| train | Jan 2017 – 30 Nov 2018 | 180,000 : 120,000 |
| validation (tuning + threshold) | 30 Nov 2018 – 12 Jan 2019 | 45,000 : 15,000 |
| test (touched once) | 12 Jan 2019 – Apr 2019 | 60,000 : 20,000 |

### Validation (threshold calibrated here to 0.5% FPR)

| Pipeline | TPR (recall) | FPR | False positives | Precision | MCC |
|---|---|---|---|---|---|
| **A: LSH + RF** | **95.47%** | 0.500% | 225 FP out of 45,000 benign | 98.45% | 0.960 |
| B: LSH + linear SVM | 74.83% | 0.500% | 225 FP out of 45,000 benign | 98.03% | 0.819 |

### Test: final A/B comparison, evaluated once (`scripts/evaluate.py`)

| Pipeline | TPR (recall) | FPR | False positives | False negatives | Precision | MCC |
|---|---|---|---|---|---|---|
| **A: LSH + RF** | **92.95%** | **0.505%** | 303 FP out of 60,000 benign | 1,411 out of 20,000 | 98.40% | 0.942 |
| B: LSH + linear SVM | 71.68% | 0.367% | 220 FP out of 60,000 benign | 5,665 out of 20,000 | 98.49% | 0.800 |

Pipeline A on test, by route:

| Route | Samples | Benign → FP | Malicious → detected |
|---|---|---|---|
| simple_malicious (lookup) | 6,014 | 7 → 7 | 6,007 → 6,007 (100%) |
| simple_benign (lookup) | 3,586 | 3,578 → 0 | 8 → 0 |
| bucket RF (7 buckets) | 18,981 | 16,673 → 15 | 2,308 → 94.7% |
| fallback: hard bucket, no RF | 48,384 | 37,525 → 262 | 10,859 → 88.9% |
| fallback: novel bucket | 3,035 | 2,217 → 19 | 818 → 90.6% |

What this shows:

* **The FPR calibration holds on later data:** 0.50% on validation and 0.505% on
  test, half of the 1% budget.
* **TPR falls from 95.5% to 92.9%** between validation (0–6 weeks after the training
  cutoff) and test (6 weeks – 4.5 months after). This is the concept drift that §8's
  30-day retraining exists for. Evaluated with a 4.5-month-old model, the test split
  is the pessimistic case.
* **The RF earns its complexity:** +21 TPR points over the linear SVM with identical
  pre-filtering. The simple regions are nearly error-free (7 FP / 9,600).
* The bucket RFs beat the fallback on their own traffic (94.7% vs 88.9%).

### Served container (`scripts/test_service.py`, `--memory=1g --cpus=1`)

| Check | Result |
|---|---|
| 536 modern benign DLL/EXE/PYD files from PyPI Windows wheels (never seen in training) | **0 false positives** |
| 300 SOREL test-period malware binaries (disarmed: machine/subsystem zeroed) | 281 detected (93.7%) |
| Latency | mean 0.07 s, p99 0.52 s; max 4.7 s for a 48 MB file (the grader's limit is 2 MiB files) |
| Memory (container) | 315 MiB idle, ~520 MiB after 836 requests (1 GiB limit) |
| Image size | ~590 MB uncompressed |

Other checks: SHAP audit, no buckets flagged (top features are import, section and
header-structure fields). Extractor fidelity is in DATA_SOURCING.md. `pytest`: 48 tests.

## Limitations

* **TPR decays with model age.** The spec's answer is the 30-day retrain cycle:
  rerun `build_dataset.py` / `fetch_features.py` / `train.py` with newer data. For
  deployment, a production cycle that also trains on the validation and test
  periods (same hyperparameters) would shrink the drift gap. It is not done here,
  because calibrating that model would need a new, more recent holdout.
* **Training data is from 2017–2019 and comes from one source** (SOREL-20M /
  ReversingLabs). The confounding check flags this. Newer malware, and benign
  software unlike SOREL's, can degrade TPR/FPR. Add sources as described in
  DATA_SOURCING.md.
* **Adversarial robustness is out of scope** (spec §14). Appending benign bytes or
  resources, or packing, can evade static features.
* LIEF 1.0 is not LIEF 0.9. The remaining extractor mismatches (DATA_SOURCING.md)
  are small but nonzero.
