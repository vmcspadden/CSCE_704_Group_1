"""Schema-mapping loader for SOREL-20M pre-extracted EMBER v2 features (§13.3).

SOREL-20M stores one EMBER v2 feature vector per sample in an LMDB database
(key = sha256, value = zlib(msgpack({0: [2381 floats]}))) on a public S3
bucket. The database is 77 GB, so rather than downloading it this module
reads the LMDB B+tree directly over HTTP range requests: meta page -> branch
pages -> leaf page -> overflow page. Branch pages are cached, so each lookup
costs roughly two 4 KiB requests.

Vectors come back on EMBER's raw scale (the LMDB holds raw values; SOREL only
log-scales them later in its data loader), i.e. exactly what
:func:`src.features.pe_parser.ember_vector` produces from a PE file.
"""
from __future__ import annotations

import bisect
import struct
import threading
import time
import zlib

import msgpack
import numpy as np
import requests

SOREL_BUCKET = "https://sorel-20m.s3.amazonaws.com/09-DEC-2020"
LMDB_URL = f"{SOREL_BUCKET}/processed-data/ember_features/data.mdb"
META_DB_URL = f"{SOREL_BUCKET}/processed-data/meta.db"

PAGE = 4096
P_BRANCH, P_LEAF, P_OVERFLOW = 0x01, 0x02, 0x04
F_BIGDATA = 0x01
EMBER_DIM = 2381


class LMDBHttpReader:
    """Read-only lookups in a remote 64-bit LMDB file (default comparator)."""

    def __init__(self, url: str = LMDB_URL, retries: int = 6):
        self.url = url
        self.retries = retries
        self._local = threading.local()
        self._branch_cache: dict[int, list[tuple[bytes, int]]] = {}
        self._lock = threading.Lock()
        self.requests_made = 0
        self.root = self._read_root()

    # -- HTTP ------------------------------------------------------------------
    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
        return s

    def _range(self, start: int, length: int) -> bytes:
        for attempt in range(self.retries):
            try:
                r = self._session().get(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"},
                                        timeout=60)
                r.raise_for_status()
                self.requests_made += 1
                return r.content
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def _page(self, pgno: int, npages: int = 1) -> bytes:
        return self._range(pgno * PAGE, npages * PAGE)

    # -- structure -------------------------------------------------------------
    def _read_root(self) -> int:
        best = None
        for pg in (0, 1):
            d = self._page(pg)
            magic, = struct.unpack_from("<I", d, 16)
            if magic != 0xBEEFC0DE:
                raise ValueError("not an LMDB file")
            # MDB_meta: magic, version, address, mapsize, dbs[2] (48 bytes each), last_pg, txnid
            main_db = 16 + 4 + 4 + 8 + 8 + 48
            root, = struct.unpack_from("<Q", d, main_db + 40)
            txnid, = struct.unpack_from("<Q", d, 16 + 24 + 96 + 8)
            if best is None or txnid > best[0]:
                best = (txnid, root)
        return best[1]

    @staticmethod
    def _nodes(d: bytes) -> list[int]:
        lower, = struct.unpack_from("<H", d, 12)
        return [struct.unpack_from("<H", d, 16 + 2 * i)[0] for i in range((lower - 16) // 2)]

    def _branch(self, pgno: int, d: bytes | None = None) -> list[tuple[bytes, int]]:
        cached = self._branch_cache.get(pgno)
        if cached is not None:
            return cached
        d = d or self._page(pgno)
        entries = []
        for off in self._nodes(d):
            lo, hi, flags, ksize = struct.unpack_from("<HHHH", d, off)
            child = lo | (hi << 16) | (flags << 32)
            entries.append((d[off + 8:off + 8 + ksize], child))
        with self._lock:
            self._branch_cache[pgno] = entries
        return entries

    def get(self, key: str) -> bytes | None:
        k = key.encode("ascii")
        pgno = self.root
        while True:
            cached = self._branch_cache.get(pgno)
            if cached is not None:
                entries = cached
            else:
                d = self._page(pgno)
                flags, = struct.unpack_from("<H", d, 10)
                if flags & P_LEAF:
                    return self._leaf_get(d, k)
                if not flags & P_BRANCH:
                    raise ValueError(f"unexpected page flags {flags:#x} at page {pgno}")
                entries = self._branch(pgno, d)
            keys = [e[0] for e in entries]
            # first key of a branch page is implicitly -infinity
            i = max(0, bisect.bisect_right(keys, k, lo=1) - 1)
            pgno = entries[i][1]

    def _leaf_get(self, d: bytes, k: bytes) -> bytes | None:
        for off in self._nodes(d):
            lo, hi, flags, ksize = struct.unpack_from("<HHHH", d, off)
            if d[off + 8:off + 8 + ksize] != k:
                continue
            size = lo | (hi << 16)
            if flags & F_BIGDATA:
                opg, = struct.unpack_from("<Q", d, off + 8 + ksize)
                npages = (16 + size + PAGE - 1) // PAGE
                return self._page(opg, npages)[16:16 + size]
            return d[off + 8 + ksize:off + 8 + ksize + size]
        return None


def decode_ember_value(raw: bytes) -> np.ndarray:
    obj = msgpack.loads(zlib.decompress(raw), strict_map_key=False)
    vec = np.asarray(obj[0], dtype=np.float32)
    if vec.shape != (EMBER_DIM,):
        raise ValueError(f"unexpected EMBER vector shape {vec.shape}")
    return vec
