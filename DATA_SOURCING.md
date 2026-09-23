# Data sourcing

## What this project trains on

**SOREL-20M** (Sophos + ReversingLabs, 2020): about 20M Windows PE samples first
seen between Jan 2017 and Apr 2019, published on a public S3 bucket
(`s3://sorel-20m/09-DEC-2020/`). It fits the design spec's requirements well:

| Spec need | SOREL-20M field |
|---|---|
| multi-engine consensus counts (§3.1) | `rl_ls_const_positives` (number of AV engines flagging the file) |
| first-seen time for the temporal split (§3.2) | `rl_fs_t` |
| static features (§4) | pre-extracted EMBER v2 vectors (LMDB, keyed by sha256) |

Its limits, and how the pipeline handles them:

* **No scan coverage count.** SOREL publishes how many engines flagged a file, not how
  many scanned it, so `min_engines_reporting` cannot be applied. `labeling.py` skips
  the coverage check when that column is empty (and logs a warning); it applies
  automatically to any source that provides it.
* **Gaps in the counts.** SOREL contains no samples with 2–4 detections. With
  `benign_consensus_threshold: 1` and `malicious_consensus_threshold: 5`, no sample is
  AMBIGUOUS. The benign threshold is 1 rather than the spec's typical 0, because a
  single engine's detection is usually noise, and excluding those files would
  drop ~4.4M benign samples, cutting benign diversity.
* **Labels are already stable.** The counts were frozen when the dataset was
  published (Dec 2020), more than 30 days after every sample's first-seen time,
  so no sample is flagged for re-verification.
* **One source.** Every sample comes from the ReversingLabs feed, so the
  §3.3 confounding report flags `sorel-20m` as the only source of both classes.
  Nothing is auto-remediated; see "Adding sources" below.
* **No benign binaries.** Only features (and disarmed malware binaries) are
  public, so training uses the pre-extracted vectors through the §13.3 schema
  loader (`src/data/sorel_loader.py`).

### How the data is fetched without downloading 100+ GB

* `meta.db` (3.8 GB sqlite) is downloaded once.
* `scripts/build_dataset.py` labels, temporally splits and samples the manifest
  (IDs and timestamps only).
* `scripts/fetch_features.py` then reads only the chosen samples' vectors
  straight out of SOREL's 77 GB LMDB file using HTTP range requests (a small
  read-only LMDB B+tree reader; about 2 × 4 KiB requests per sample, ~180
  samples/s).

SOREL's `lightGBM-features/*.npz` files are *not* used: their rows are
shuffled, so a row cannot be mapped back to its sha256 or first-seen time.

### Feature compatibility (why `pe_parser.py` looks the way it does)

The SOREL vectors were computed with EMBER v2 on LIEF 0.9. `src/features/pe_parser.py`
re-implements EMBER v2 on LIEF 1.0 and reproduces the LIEF 0.9 quirks that affect
the vector: character-wise hashing of the entry-section name, backslash-escaped
non-UTF-8 names, and ordinal→name resolution tables in the import count.
`scripts/check_extractor_fidelity.py` compares the output with SOREL's stored
vectors on SOREL's own (disarmed) binaries:

| block | samples with any mismatch (n=300) |
|---|---|
| histogram, strings, header*, imports, data directories | 0–0.3% |
| section | 1% |
| general | 4% |
| exports | 6% |
| byte-entropy histogram | 21% (mostly from the zeroed header bytes shifting entropy windows) |

\* excluding the machine/subsystem bins, which SOREL zeroed to disarm the binaries.

## Adding sources

To reduce single-source concentration, add a manifest with the same columns
(`sample_id, first_seen_utc, engines_malicious, engines_reporting,
consensus_computed_at_utc, source`) and either EMBER v2 vectors or raw binaries
(vectorized with `pe_parser.ember_vector`). Candidates:

* **EMBER 2018** (elastic): raw EMBER JSON with the `appeared` month, the same
  feature schema.
* **BODMAS** (2019–2020): EMBER v2 vectors plus timestamps and family labels.
* **VirusTotal** feeds (or academic access) for fresh labeled binaries with full
  engine coverage.
* Benign binaries you have the right to use (for example Windows install media or
  signed vendor software) to widen benign diversity.

Never add VirusTotal detection counts or history as model *features*. They are
labels only (§4.1).

## Out-of-distribution benign check

`scripts/collect_benign_pe.py` downloads ~540 DLL/EXE/PYD files from Windows wheels
of well-known PyPI projects. They are modern (2024–2026 builds) and never used for
training or calibration. They are only used to sanity-check the served model's
false-positive rate on software the training data cannot contain.
