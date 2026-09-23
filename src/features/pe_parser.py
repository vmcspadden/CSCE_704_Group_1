"""Raw PE attribute extraction, EMBER v2 compatible.

The training data (SOREL-20M) ships EMBER feature-version-2 vectors computed
with LIEF 0.9. To make live PE files land in the same feature space, this
module re-implements EMBER v2's raw-feature and vectorization logic
(github.com/elastic/ember, features.py) on top of a current LIEF release.
LIEF is used here (not pefile) because the training vectors were produced
with LIEF, so its section entropy, virtual size, import and export parsing
match the training data most closely.

LIEF's Python enum names have changed since 0.9, so all names that feed the
feature hasher (machine, subsystem, characteristics, section properties) are
produced from the raw integer fields with the LIEF 0.9 name tables below.
Two LIEF 0.9 behaviours are reproduced deliberately:
  * entry section: 0.9 looked the entry point *VA* up as a file offset,
    which almost never matches, so it fell back to the first executable
    section; ``_entry_section`` does the same.
  * section properties: every enum value v with ``characteristics & v != 0``
    is reported (this also yields several ALIGN_* names for one value).

Layout of the 2381-float vector (EMBER v2 order):
  histogram 256 | byteentropy 256 | strings 104 | general 10 | header 62 |
  section 255 | imports 1280 | exports 128 | datadirectories 30
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import lief
import numpy as np
from sklearn.feature_extraction import FeatureHasher

lief.logging.disable()

EMBER_DIM = 2381
BLOCKS = [("histogram", 256), ("byteentropy", 256), ("strings", 104), ("general", 10), ("header", 62),
          ("section", 255), ("imports", 1280), ("exports", 128), ("datadirectories", 30)]
OFFSETS: dict[str, tuple[int, int]] = {}
_o = 0
for _name, _dim in BLOCKS:
    OFFSETS[_name] = (_o, _o + _dim)
    _o += _dim
assert _o == EMBER_DIM

# --- LIEF 0.9 name tables --------------------------------------------------------
MACHINE = {0x0: "UNKNOWN", 0x1d3: "AM33", 0x8664: "AMD64", 0x1c0: "ARM", 0x1c4: "ARMNT", 0xaa64: "ARM64",
           0xebc: "EBC", 0x14c: "I386", 0x200: "IA64", 0x9041: "M32R", 0x266: "MIPS16", 0x366: "MIPSFPU",
           0x466: "MIPSFPU16", 0x1f0: "POWERPC", 0x1f1: "POWERPCFP", 0x166: "R4000", 0x1a2: "SH3",
           0x1a3: "SH3DSP", 0x1a6: "SH4", 0x1a8: "SH5", 0x1c2: "THUMB", 0x169: "WCEMIPSV2"}
SUBSYSTEM = {0: "UNKNOWN", 1: "NATIVE", 2: "WINDOWS_GUI", 3: "WINDOWS_CUI", 5: "OS2_CUI", 7: "POSIX_CUI",
             8: "NATIVE_WINDOWS", 9: "WINDOWS_CE_GUI", 10: "EFI_APPLICATION", 11: "EFI_BOOT_SERVICE_DRIVER",
             12: "EFI_RUNTIME_DRIVER", 13: "EFI_ROM", 14: "XBOX", 16: "WINDOWS_BOOT_APPLICATION"}
HEADER_CHARACTERISTICS = [
    (0x0001, "RELOCS_STRIPPED"), (0x0002, "EXECUTABLE_IMAGE"), (0x0004, "LINE_NUMS_STRIPPED"),
    (0x0008, "LOCAL_SYMS_STRIPPED"), (0x0010, "AGGRESSIVE_WS_TRIM"), (0x0020, "LARGE_ADDRESS_AWARE"),
    (0x0080, "BYTES_REVERSED_LO"), (0x0100, "CHARA_32BIT_MACHINE"), (0x0200, "DEBUG_STRIPPED"),
    (0x0400, "REMOVABLE_RUN_FROM_SWAP"), (0x0800, "NET_RUN_FROM_SWAP"), (0x1000, "SYSTEM"), (0x2000, "DLL"),
    (0x4000, "UP_SYSTEM_ONLY"), (0x8000, "BYTES_REVERSED_HI")]
DLL_CHARACTERISTICS = [
    (0x0020, "HIGH_ENTROPY_VA"), (0x0040, "DYNAMIC_BASE"), (0x0080, "FORCE_INTEGRITY"), (0x0100, "NX_COMPAT"),
    (0x0200, "NO_ISOLATION"), (0x0400, "NO_SEH"), (0x0800, "NO_BIND"), (0x1000, "APPCONTAINER"),
    (0x2000, "WDM_DRIVER"), (0x4000, "GUARD_CF"), (0x8000, "TERMINAL_SERVER_AWARE")]
SECTION_CHARACTERISTICS = [
    (0x00000008, "TYPE_NO_PAD"), (0x00000020, "CNT_CODE"), (0x00000040, "CNT_INITIALIZED_DATA"),
    (0x00000080, "CNT_UNINITIALIZED_DATA"), (0x00000100, "LNK_OTHER"), (0x00000200, "LNK_INFO"),
    (0x00000800, "LNK_REMOVE"), (0x00001000, "LNK_COMDAT"), (0x00008000, "GPREL"),
    (0x00020000, "MEM_PURGEABLE"), (0x00040000, "MEM_LOCKED"), (0x00080000, "MEM_PRELOAD"),
    (0x00100000, "ALIGN_1BYTES"), (0x00200000, "ALIGN_2BYTES"), (0x00300000, "ALIGN_4BYTES"),
    (0x00400000, "ALIGN_8BYTES"), (0x00500000, "ALIGN_16BYTES"), (0x00600000, "ALIGN_32BYTES"),
    (0x00700000, "ALIGN_64BYTES"), (0x00800000, "ALIGN_128BYTES"), (0x00900000, "ALIGN_256BYTES"),
    (0x00A00000, "ALIGN_512BYTES"), (0x00B00000, "ALIGN_1024BYTES"), (0x00C00000, "ALIGN_2048BYTES"),
    (0x00D00000, "ALIGN_4096BYTES"), (0x00E00000, "ALIGN_8192BYTES"), (0x01000000, "LNK_NRELOC_OVFL"),
    (0x02000000, "MEM_DISCARDABLE"), (0x04000000, "MEM_NOT_CACHED"), (0x08000000, "MEM_NOT_PAGED"),
    (0x10000000, "MEM_SHARED"), (0x20000000, "MEM_EXECUTE"), (0x40000000, "MEM_READ"),
    (0x80000000, "MEM_WRITE")]
MEM_EXECUTE = 0x20000000
# Ordinal->name tables LIEF 0.9 used when counting imported functions
# (LIEF 0.9 src/PE/utils/ordinals_lookup_tables); only the ordinals are kept.
with open(Path(__file__).with_name("lief09_ordinals.json"), encoding="utf-8") as _fh:
    LIEF09_ORDINALS = {lib: frozenset(ords) for lib, ords in json.load(_fh).items()}

_STRINGS = re.compile(b"[\x20-\x7f]{5,}")
_PATHS = re.compile(b"c:\\\\", re.IGNORECASE)
_URLS = re.compile(b"https?://", re.IGNORECASE)
_REGISTRY = re.compile(b"HKEY_")
_MZ = re.compile(b"MZ")


def _flags(value: int, table) -> list[str]:
    return [name for bit, name in table if value & bit]


def _str(name) -> str:
    """LIEF >= 0.12 returns bytes for names that are not valid UTF-8; LIEF 0.9
    decoded them with backslash escapes, which is what the training data holds."""
    if isinstance(name, (bytes, bytearray)):
        return bytes(name).decode("utf-8", "backslashreplace")
    return name


def _hash(values, n: int, input_type: str = "string") -> np.ndarray:
    return FeatureHasher(n, input_type=input_type).transform([values]).toarray()[0]


def parse_binary(bytez: bytes):
    """Parse with LIEF; None when the file is not a parseable PE (EMBER then
    zero-fills every parsed feature, exactly as its own extractor does)."""
    try:
        binary = lief.PE.parse(list(bytez) if isinstance(bytez, (bytes, bytearray)) else bytez)
    except Exception:  # noqa: BLE001 - any parser failure means "unparseable"
        return None
    return binary if isinstance(binary, lief.PE.Binary) else None


# --- raw features (EMBER JSON schema) ----------------------------------------------
def byte_histogram(bytez: bytes) -> list[int]:
    return np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256).tolist()


def byte_entropy_histogram(bytez: bytes, step: int = 1024, window: int = 2048) -> list[int]:
    """EMBER's windowed byte/entropy histogram, vectorized.

    EMBER slides a 2048-byte window in 1024-byte steps, so every window is two
    adjacent 1024-byte chunks: per-chunk nibble counts are computed once and
    summed pairwise (identical result, O(n) instead of a Python loop)."""
    output = np.zeros((16, 16), dtype=np.int64)
    a = np.frombuffer(bytez, dtype=np.uint8)
    if a.shape[0] < window:
        if a.size:
            c = np.bincount(a >> 4, minlength=16)
            output[_entropy_bins(c[None, :], window)[0], :] += c
        return output.flatten().tolist()
    assert window == 2 * step
    n_windows = (a.shape[0] - window) // step + 1
    chunks = (a[:(n_windows + 1) * step] >> 4).reshape(n_windows + 1, step)
    offsets = (np.arange(n_windows + 1) * 16)[:, None]
    chunk_counts = np.bincount((chunks + offsets).ravel(), minlength=(n_windows + 1) * 16).reshape(-1, 16)
    counts = chunk_counts[:-1] + chunk_counts[1:]
    bins = _entropy_bins(counts, window)
    np.add.at(output, bins, counts)
    return output.flatten().tolist()


def _entropy_bins(counts: np.ndarray, window: int) -> np.ndarray:
    p = counts.astype(np.float32) / window
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(counts > 0, -p * np.log2(p), 0.0)
    H = terms.sum(axis=1) * 2
    return np.minimum((H * 2).astype(np.int64), 15)


def string_features(bytez: bytes) -> dict:
    allstrings = _STRINGS.findall(bytez)
    if allstrings:
        lengths = [len(s) for s in allstrings]
        avlength = sum(lengths) / len(lengths)
        shifted = np.frombuffer(b"".join(allstrings), dtype=np.uint8) - 0x20
        c = np.bincount(shifted, minlength=96)
        csum = c.sum()
        p = c.astype(np.float32) / csum
        wh = np.where(c)[0]
        H = float(np.sum(-p[wh] * np.log2(p[wh])))
    else:
        avlength, c, H, csum = 0, np.zeros(96, dtype=np.float32), 0.0, 0
    return {"numstrings": len(allstrings), "avlength": avlength, "printabledist": c.tolist(),
            "printables": int(csum), "entropy": H, "paths": len(_PATHS.findall(bytez)),
            "urls": len(_URLS.findall(bytez)), "registry": len(_REGISTRY.findall(bytez)),
            "MZ": len(_MZ.findall(bytez))}


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - malformed fields fall back like EMBER's zero-fill
        return default


def general_features(bytez: bytes, b) -> dict:
    if b is None:
        return {"size": len(bytez), "vsize": 0, "has_debug": 0, "exports": 0, "imports": 0,
                "has_relocations": 0, "has_resources": 0, "has_signature": 0, "has_tls": 0, "symbols": 0}
    return {
        "size": len(bytez),
        "vsize": _safe(lambda: int(b.virtual_size), 0),
        "has_debug": _safe(lambda: int(b.has_debug), 0),
        "exports": len(_named_exports(b)),
        "imports": _imported_function_count(b),
        "has_relocations": _safe(lambda: int(b.has_relocations), 0),
        "has_resources": _safe(lambda: int(b.has_resources), 0),
        "has_signature": _safe(lambda: int(b.has_signatures), 0),
        "has_tls": _safe(lambda: int(b.has_tls), 0),
        "symbols": _safe(lambda: len(b.symbols), 0),
    }


def header_features(b) -> dict:
    raw = {"coff": {"timestamp": 0, "machine": "", "characteristics": []},
           "optional": {"subsystem": "", "dll_characteristics": [], "magic": "", "major_image_version": 0,
                        "minor_image_version": 0, "major_linker_version": 0, "minor_linker_version": 0,
                        "major_operating_system_version": 0, "minor_operating_system_version": 0,
                        "major_subsystem_version": 0, "minor_subsystem_version": 0, "sizeof_code": 0,
                        "sizeof_headers": 0, "sizeof_heap_commit": 0}}
    if b is None:
        return raw
    h, o = b.header, b.optional_header
    raw["coff"]["timestamp"] = int(h.time_date_stamps)
    raw["coff"]["machine"] = MACHINE.get(int(h.machine), "UNKNOWN")
    raw["coff"]["characteristics"] = _flags(int(h.characteristics), HEADER_CHARACTERISTICS)
    opt = raw["optional"]
    opt["subsystem"] = SUBSYSTEM.get(int(o.subsystem), "UNKNOWN")
    opt["dll_characteristics"] = _flags(int(o.dll_characteristics), DLL_CHARACTERISTICS)
    opt["magic"] = "PE32_PLUS" if int(o.magic) == 0x20B else "PE32"
    for k in ("major_image_version", "minor_image_version", "major_linker_version", "minor_linker_version",
              "major_operating_system_version", "minor_operating_system_version", "major_subsystem_version",
              "minor_subsystem_version", "sizeof_code", "sizeof_headers", "sizeof_heap_commit"):
        opt[k] = int(getattr(o, k))
    return raw


def _entry_section(b, sections) -> str:
    # LIEF 0.9: section_from_offset(entrypoint VA); on failure, first MEM_EXECUTE section.
    ep = _safe(lambda: int(b.entrypoint), -1)
    for s in sections:
        start, size = int(s.pointerto_raw_data), int(s.sizeof_raw_data)
        if start <= ep < start + size:
            return _str(s.name)
    for s in sections:
        if int(s.characteristics) & MEM_EXECUTE:
            return _str(s.name)
    return ""


def section_features(b) -> dict:
    if b is None:
        return {"entry": "", "sections": []}
    sections = _safe(lambda: list(b.sections), [])
    return {
        "entry": _entry_section(b, sections),
        "sections": [{"name": _str(s.name), "size": int(s.sizeof_raw_data), "entropy": float(s.entropy),
                      "vsize": int(s.virtual_size),
                      "props": _flags(int(s.characteristics), SECTION_CHARACTERISTICS)} for s in sections],
    }


def import_features(b) -> dict:
    imports: dict[str, list[str]] = {}
    if b is None:
        return imports
    for lib in _safe(lambda: list(b.imports), []):
        entries = imports.setdefault(_str(lib.name), [])
        for e in _safe(lambda: list(lib.entries), []):
            entries.append(f"ordinal{e.ordinal}" if e.is_ordinal else _str(e.name)[:10000])
    return imports


def _imported_function_count(b) -> int:
    """LIEF 0.9 len(imported_functions): named entries, plus ordinal
    entries of libraries it had ordinal->name tables for."""
    count = 0
    for lib in _safe(lambda: list(b.imports), []):
        table = LIEF09_ORDINALS.get(_str(lib.name).lower(), frozenset())
        for e in _safe(lambda: list(lib.entries), []):
            count += (e.ordinal in table) if e.is_ordinal else bool(e.name)
    return count


def _named_exports(b) -> list[str]:
    exp = _safe(lambda: b.get_export(), None)
    if exp is None:
        return []
    return [_str(e.name)[:10000] for e in _safe(lambda: list(exp.entries), []) if e.name]


def data_directories(b) -> list[dict]:
    if b is None:
        return []
    return [{"size": int(d.size), "virtual_address": int(d.rva)}
            for d in _safe(lambda: list(b.data_directories), [])]


def raw_features(bytez: bytes, binary=None, parse: bool = True) -> dict:
    """EMBER v2 raw-feature dict for a PE file (JSON-serializable)."""
    b = binary if binary is not None else (parse_binary(bytez) if parse else None)
    return {
        "histogram": byte_histogram(bytez),
        "byteentropy": byte_entropy_histogram(bytez),
        "strings": string_features(bytez),
        "general": general_features(bytez, b),
        "header": header_features(b),
        "section": section_features(b),
        "imports": import_features(b),
        "exports": _named_exports(b) if b is not None else [],
        "datadirectories": data_directories(b),
        "parsed": b is not None,
    }


# --- vectorization (EMBER v2 process_raw_features) ------------------------------
def vectorize(raw: dict) -> np.ndarray:
    hist = np.asarray(raw["histogram"], dtype=np.float32)
    hist = hist / max(hist.sum(), 1)
    be = np.asarray(raw["byteentropy"], dtype=np.float32)
    be = be / max(be.sum(), 1)

    s = raw["strings"]
    divisor = float(s["printables"]) if s["printables"] > 0 else 1.0
    strings = np.hstack([s["numstrings"], s["avlength"], s["printables"],
                         np.asarray(s["printabledist"]) / divisor, s["entropy"], s["paths"], s["urls"],
                         s["registry"], s["MZ"]])

    g = raw["general"]
    general = [g["size"], g["vsize"], g["has_debug"], g["exports"], g["imports"], g["has_relocations"],
               g["has_resources"], g["has_signature"], g["has_tls"], g["symbols"]]

    h = raw["header"]
    o = h["optional"]
    header = np.hstack([
        h["coff"]["timestamp"],
        _hash([h["coff"]["machine"]], 10), _hash(h["coff"]["characteristics"], 10),
        _hash([o["subsystem"]], 10), _hash(o["dll_characteristics"], 10), _hash([o["magic"]], 10),
        o["major_image_version"], o["minor_image_version"], o["major_linker_version"], o["minor_linker_version"],
        o["major_operating_system_version"], o["minor_operating_system_version"], o["major_subsystem_version"],
        o["minor_subsystem_version"], o["sizeof_code"], o["sizeof_headers"], o["sizeof_heap_commit"]])

    sec = raw["section"]
    sections = sec["sections"]
    sec_general = [len(sections), sum(1 for x in sections if x["size"] == 0),
                   sum(1 for x in sections if x["name"] == ""),
                   sum(1 for x in sections if "MEM_READ" in x["props"] and "MEM_EXECUTE" in x["props"]),
                   sum(1 for x in sections if "MEM_WRITE" in x["props"])]
    section = np.hstack([
        sec_general,
        _hash([(x["name"], x["size"]) for x in sections], 50, "pair"),
        _hash([(x["name"], x["entropy"]) for x in sections], 50, "pair"),
        _hash([(x["name"], x["vsize"]) for x in sections], 50, "pair"),
        # EMBER passes the bare string, so FeatureHasher hashes its characters.
        _hash(list(sec["entry"]), 50),
        _hash([p for x in sections for p in x["props"] if x["name"] == sec["entry"]], 50)])

    imp = raw["imports"]
    libraries = list({lib.lower() for lib in imp})
    functions = [lib.lower() + ":" + e for lib, entries in imp.items() for e in entries]
    imports = np.hstack([_hash(libraries, 256), _hash(functions, 1024)])
    exports = _hash(raw["exports"], 128)

    dd = np.zeros(30, dtype=np.float32)
    for i, d in enumerate(raw["datadirectories"][:15]):
        dd[2 * i] = d["size"]
        dd[2 * i + 1] = d["virtual_address"]

    vec = np.hstack([hist, be, strings, general, header, section, imports, exports, dd]).astype(np.float32)
    assert vec.shape == (EMBER_DIM,), vec.shape
    return vec


def ember_vector(bytez: bytes) -> tuple[np.ndarray, bool]:
    """PE bytes -> (EMBER v2 vector, parsed_ok)."""
    raw = raw_features(bytez)
    return vectorize(raw), raw["parsed"]
