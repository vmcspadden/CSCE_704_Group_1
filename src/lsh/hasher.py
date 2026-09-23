"""§5 MinHash LSH pre-filter with multiple feature-subset mappings.

Each sample is turned into a set of discrete tokens per mapping:

  structure_focused  header fields (machine, subsystem, characteristics,
                     linker/OS/subsystem versions), section-name and
                     entry-section tokens, imported-library and imported
                     function tokens, export presence
  content_focused    string statistics, resource size/entropy, coarse byte
                     entropy profile, file size

A MinHash signature of ``rows_per_band`` values is computed per mapping and
the band is hashed into a bucket id. Near-identical files (malware variants
with small mutations) share most tokens and therefore land in the same bucket
with high probability, which a cryptographic hash would not give.

Only the random permutation parameters are "fitted" (drawn from ``seed``),
so a mapping has no data-dependent state: querying it on validation/test
samples can never leak their information into the mapping (§5.1). Bucket
*contents* are built from the training split only, in bucket_manager.py.

§5.2 decision: option (a). The ``primary_mapping`` (structure_focused) alone
assigns buckets and classifiers; content_focused bucket ids are computed and
reported as an auxiliary signal.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from src.features.feature_extractor import _RSRC_BIN
from src.features.pe_parser import OFFSETS

MERSENNE = (1 << 61) - 1
_S = OFFSETS["section"][0]
_H = OFFSETS["header"][0]
_G = OFFSETS["general"][0]
_I = OFFSETS["imports"][0]
_X = OFFSETS["exports"][0]
_T = OFFSETS["strings"][0]
_B = OFFSETS["byteentropy"][0]
_D = OFFSETS["datadirectories"][0]


def _log2_bucket(v: float) -> int:
    return int(np.log2(1.0 + max(0.0, float(v))))


def _nz(row: np.ndarray, start: int, n: int, prefix: str) -> list[str]:
    seg = row[start:start + n]
    idx = np.flatnonzero(seg)
    return [f"{prefix}{i}{'+' if seg[i] > 0 else '-'}" for i in idx]


def structure_tokens(e: np.ndarray) -> list[str]:
    toks = []
    toks += _nz(e, _H + 1, 10, "machine:")
    toks += _nz(e, _H + 11, 10, "chars:")
    toks += _nz(e, _H + 21, 10, "subsys:")
    toks += _nz(e, _H + 31, 10, "dllchars:")
    toks += _nz(e, _H + 41, 10, "magic:")
    v = e[_H + 51:_H + 62]
    toks.append(f"linker:{int(v[2])}.{int(v[3])}")
    toks.append(f"os:{int(v[4])}.{int(v[5])}")
    toks.append(f"subsysver:{int(v[6])}.{int(v[7])}")
    toks.append(f"nsec:{int(e[_S])}")
    toks += _nz(e, _S + 5 + 100, 50, "secname:")        # hashed section vsize -> names present
    toks += _nz(e, _S + 5 + 150, 50, "entryname:")
    toks += _nz(e, _S + 5 + 200, 50, "entryprops:")
    toks += _nz(e, _I, 256, "lib:")
    toks += _nz(e, _I + 256, 1024, "imp:")
    toks.append(f"nimp:{_log2_bucket(e[_G + 4])}")
    toks.append(f"nexp:{_log2_bucket(e[_G + 3])}")
    toks.append(f"flags:{int(e[_G + 2] > 0)}{int(e[_G + 5] > 0)}{int(e[_G + 6] > 0)}{int(e[_G + 8] > 0)}")
    return toks


def content_tokens(e: np.ndarray) -> list[str]:
    toks = [f"size:{_log2_bucket(e[_G])}", f"nstr:{_log2_bucket(e[_T])}", f"avlen:{int(min(e[_T + 1], 200) // 4)}",
            f"strent:{int(e[_T + 99] * 4)}", f"paths:{_log2_bucket(e[_T + 100])}", f"urls:{_log2_bucket(e[_T + 101])}",
            f"reg:{_log2_bucket(e[_T + 102])}", f"mz:{_log2_bucket(e[_T + 103])}",
            f"rsrc:{_log2_bucket(e[_D + 4])}", f"rsrcent:{int(abs(e[_S + 55 + _RSRC_BIN]) * 2)}",
            f"sig:{int(e[_G + 7] > 0)}"]
    be = e[_B:_B + 256].reshape(16, 16).sum(axis=1)  # mass per entropy level
    total = be.sum()
    if total > 0:
        for lvl in np.flatnonzero(be / total > 0.05):
            toks.append(f"entlvl:{lvl}:{int(10 * be[lvl] / total)}")
    return toks


TOKENIZERS = {"structure_focused": structure_tokens, "content_focused": content_tokens}


@dataclass
class MinHashMapping:
    name: str
    rows_per_band: int
    seed: int
    a: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.name not in TOKENIZERS:
            raise ValueError(f"unknown mapping {self.name}")
        rng = np.random.default_rng(self.seed)
        self.a = rng.integers(1, MERSENNE, size=self.rows_per_band, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, size=self.rows_per_band, dtype=np.uint64)

    def signature(self, ember_row: np.ndarray) -> np.ndarray:
        toks = TOKENIZERS[self.name](ember_row)
        if not toks:
            return np.full(self.rows_per_band, MERSENNE, dtype=np.uint64)
        x = np.fromiter((zlib.crc32(t.encode()) for t in toks), dtype=np.uint64, count=len(toks))
        # (a*x + b) mod p on 64-bit ints: x < 2^32, a < 2^61 would overflow, so
        # split a into 30-bit halves (both partial products stay < 2^62).
        a_hi, a_lo = self.a >> np.uint64(30), self.a & np.uint64((1 << 30) - 1)
        hi = (a_hi[:, None] * x[None, :]) % np.uint64(MERSENNE)
        hi = (hi * np.uint64(1 << 30)) % np.uint64(MERSENNE)
        lo = (a_lo[:, None] * x[None, :]) % np.uint64(MERSENNE)
        h = (hi + lo + self.b[:, None]) % np.uint64(MERSENNE)
        return h.min(axis=1)

    def bucket(self, ember_row: np.ndarray) -> str:
        sig = self.signature(ember_row)
        return f"{self.name[0]}{zlib.crc32(sig.tobytes()):08x}"


class LSHPrefilter:
    """All mappings; the primary one decides bucket membership (§5.2 option a)."""

    def __init__(self, cfg):
        self.primary = cfg["primary_mapping"]
        self.mappings = {name: MinHashMapping(name, m["rows_per_band"], cfg["seed"] + i)
                         for i, (name, m) in enumerate(sorted(cfg["mappings"].items()))}
        if self.primary not in self.mappings:
            raise ValueError(f"primary mapping {self.primary} not configured")

    def buckets(self, ember: np.ndarray, mapping: str | None = None) -> np.ndarray:
        m = self.mappings[mapping or self.primary]
        E = np.atleast_2d(ember)
        return np.array([m.bucket(row) for row in E], dtype=object)

    def all_buckets(self, ember: np.ndarray) -> dict[str, np.ndarray]:
        return {name: self.buckets(ember, name) for name in self.mappings}
