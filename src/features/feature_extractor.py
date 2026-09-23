"""Versioned vectorization: EMBER v2 vector -> named model features (§4).

Input is always an EMBER v2 vector (raw scale) - computed from a PE file by
:mod:`src.features.pe_parser` at inference time, or loaded from SOREL-20M by
:mod:`src.data.sorel_loader` for training. Both paths go through this module,
so training and serving features are identical by construction (§13.3).

Every output feature has a name and one §4.1 category:

  header      COFF/optional header fields, file/virtual size, 32/64-bit
  sections    section counts, hashed per-section size/entropy/vsize,
              entry-section name and properties
  imports     hashed libraries and library:function pairs, import/export
              counts, hashed exports, import-related data directories, and a
              fixed vocabulary of suspicious APIs (per-category scores)
  resources   has_resources, resource table size, .rsrc size/entropy
  strings     count, mean length, printable distribution, URL / registry /
              path / MZ counts
  signature   Authenticode presence and certificate table size
  bytes       byte histogram and byte/entropy histogram

``resources_entry_count`` and ``header_entropy`` (on the §9 SHAP watch-list)
are not derivable from the EMBER v2 schema, so they do not exist in this
extractor version; the SHAP audit reports them as absent.

Numeric values go through a signed log1p (monotone, so tree models are
unaffected; the linear SVM baseline needs it). Feature vectors from
different ``EXTRACTOR_VERSION``s must never be mixed (§4.2): stored feature
matrices carry the version, and :func:`check_version` refuses mismatches.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from src.features.pe_parser import EMBER_DIM, OFFSETS

EXTRACTOR_VERSION = "1.0.0"

DATA_DIRECTORIES = ["EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
                    "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE",
                    "LOAD_CONFIG_TABLE", "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"]
GENERAL = ["size", "vsize", "has_debug", "exports", "imports", "has_relocations", "has_resources",
           "has_signature", "has_tls", "symbols"]
HEADER_TAIL = ["major_image_version", "minor_image_version", "major_linker_version", "minor_linker_version",
               "major_operating_system_version", "minor_operating_system_version", "major_subsystem_version",
               "minor_subsystem_version", "sizeof_code", "sizeof_headers", "sizeof_heap_commit"]
STRINGS_HEAD = ["numstrings", "avlength", "printables"]
STRINGS_TAIL = ["entropy", "paths", "urls", "registry", "MZ"]

# Fixed vocabulary of suspicious / commonly abused APIs (§4.1). EMBER hashes
# "library:function" pairs into 1024 bins, so each API is located by its bin;
# a category score counts vocabulary bins that are occupied (collisions with
# other imports are possible - this is an approximation by construction).
SUSPICIOUS_APIS = {
    "file_io": ["kernel32.dll:CreateFileA", "kernel32.dll:CreateFileW", "kernel32.dll:WriteFile",
                "kernel32.dll:DeleteFileA", "kernel32.dll:DeleteFileW", "kernel32.dll:MoveFileExW",
                "kernel32.dll:CopyFileA", "kernel32.dll:CopyFileW", "kernel32.dll:FindFirstFileA",
                "kernel32.dll:FindFirstFileW", "kernel32.dll:SetFileAttributesW"],
    "registry": ["advapi32.dll:RegOpenKeyExA", "advapi32.dll:RegOpenKeyExW", "advapi32.dll:RegSetValueExA",
                 "advapi32.dll:RegSetValueExW", "advapi32.dll:RegCreateKeyExA", "advapi32.dll:RegCreateKeyExW",
                 "advapi32.dll:RegDeleteKeyA", "advapi32.dll:RegDeleteValueA"],
    "process": ["kernel32.dll:CreateProcessA", "kernel32.dll:CreateProcessW", "kernel32.dll:OpenProcess",
                "kernel32.dll:TerminateProcess", "kernel32.dll:WinExec", "shell32.dll:ShellExecuteA",
                "shell32.dll:ShellExecuteW", "shell32.dll:ShellExecuteExW", "kernel32.dll:CreateToolhelp32Snapshot",
                "kernel32.dll:Process32First", "kernel32.dll:Process32Next", "advapi32.dll:CreateServiceA",
                "advapi32.dll:StartServiceA"],
    "network": ["ws2_32.dll:connect", "ws2_32.dll:send", "ws2_32.dll:recv", "ws2_32.dll:WSAStartup",
                "wininet.dll:InternetOpenA", "wininet.dll:InternetOpenUrlA", "wininet.dll:InternetReadFile",
                "wininet.dll:HttpSendRequestA", "urlmon.dll:URLDownloadToFileA", "urlmon.dll:URLDownloadToFileW",
                "winhttp.dll:WinHttpOpen", "ws2_32.dll:gethostbyname"],
    "injection": ["kernel32.dll:VirtualAllocEx", "kernel32.dll:WriteProcessMemory",
                  "kernel32.dll:CreateRemoteThread", "kernel32.dll:VirtualProtectEx", "kernel32.dll:VirtualAlloc",
                  "kernel32.dll:VirtualProtect", "ntdll.dll:NtUnmapViewOfSection", "ntdll.dll:ZwUnmapViewOfSection",
                  "kernel32.dll:SetThreadContext", "kernel32.dll:ResumeThread", "kernel32.dll:QueueUserAPC",
                  "user32.dll:SetWindowsHookExA", "user32.dll:SetWindowsHookExW", "kernel32.dll:LoadLibraryA",
                  "kernel32.dll:GetProcAddress"],
    "anti_debug": ["kernel32.dll:IsDebuggerPresent", "kernel32.dll:CheckRemoteDebuggerPresent",
                   "ntdll.dll:NtQueryInformationProcess", "kernel32.dll:OutputDebugStringA",
                   "kernel32.dll:GetTickCount", "kernel32.dll:QueryPerformanceCounter", "kernel32.dll:Sleep"],
    "crypto": ["advapi32.dll:CryptEncrypt", "advapi32.dll:CryptDecrypt", "advapi32.dll:CryptAcquireContextA",
               "advapi32.dll:CryptAcquireContextW", "advapi32.dll:CryptGenKey", "advapi32.dll:CryptImportKey",
               "bcrypt.dll:BCryptEncrypt", "crypt32.dll:CryptStringToBinaryA"],
    "keylog_capture": ["user32.dll:GetAsyncKeyState", "user32.dll:GetKeyState", "user32.dll:GetForegroundWindow",
                       "gdi32.dll:BitBlt", "user32.dll:GetClipboardData"],
}


def _bin(token: str, n: int) -> tuple[int, float]:
    row = FeatureHasher(n, input_type="string").transform([[token]])
    return int(row.indices[0]), float(row.data[0])


def _sections_bin(name: str) -> tuple[int, float]:
    row = FeatureHasher(50, input_type="pair").transform([[(name, 1.0)]])
    return int(row.indices[0]), float(row.data[0])


def _build_names() -> tuple[list[str], list[str]]:
    names, cats = [], []

    def add(n, c):
        names.append(n)
        cats.append(c)

    for i in range(256):
        add(f"bytes_hist_{i:03d}", "bytes")
    for i in range(256):
        add(f"bytes_entropy_hist_{i:03d}", "bytes")
    for n in STRINGS_HEAD:
        add(f"strings_{n}", "strings")
    for i in range(96):
        add(f"strings_printable_{i:02d}", "strings")
    for n in STRINGS_TAIL:
        add(f"strings_{n}", "strings")
    general_cat = {"size": "header", "vsize": "header", "has_debug": "header", "exports": "imports",
                   "imports": "imports", "has_relocations": "header", "has_resources": "resources",
                   "has_signature": "signature", "has_tls": "header", "symbols": "header"}
    for n in GENERAL:
        add(f"{general_cat[n]}_{n}", general_cat[n])
    add("header_timestamp", "header")
    for field in ["machine", "characteristics", "subsystem", "dll_characteristics", "magic"]:
        for i in range(10):
            add(f"header_{field}_h{i}", "header")
    for n in HEADER_TAIL:
        add(f"header_{n}", "header")
    for n in ["count", "zero_size", "empty_name", "rx", "w"]:
        add(f"sections_{n}", "sections")
    for field in ["size", "entropy", "vsize", "entry_name", "entry_props"]:
        for i in range(50):
            add(f"sections_{field}_h{i:02d}", "sections")
    for i in range(256):
        add(f"imports_lib_h{i:03d}", "imports")
    for i in range(1024):
        add(f"imports_func_h{i:04d}", "imports")
    for i in range(128):
        add(f"imports_export_h{i:03d}", "imports")
    dd_cat = {"EXPORT_TABLE": "imports", "IMPORT_TABLE": "imports", "IAT": "imports",
              "DELAY_IMPORT_DESCRIPTOR": "imports", "RESOURCE_TABLE": "resources",
              "CERTIFICATE_TABLE": "signature"}
    for d in DATA_DIRECTORIES:
        c = dd_cat.get(d, "header")
        add(f"{c}_dd_{d.lower()}_size", c)
        add(f"{c}_dd_{d.lower()}_rva", c)
    assert len(names) == EMBER_DIM
    # derived features
    for n in ["resources_entropy", "resources_size", "resources_vsize"]:
        add(n, "resources")
    add("header_is_64bit", "header")
    add("signature_present", "signature")
    for cat in SUSPICIOUS_APIS:
        add(f"imports_suspicious_{cat}", "imports")
    return names, cats


FEATURE_NAMES, FEATURE_CATEGORIES = _build_names()
N_FEATURES = len(FEATURE_NAMES)
CATEGORIES = sorted(set(FEATURE_CATEGORIES))
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

_RSRC_BIN, _RSRC_SIGN = _sections_bin(".rsrc")
_MAGIC_PLUS_BIN, _MAGIC_PLUS_SIGN = _bin("PE32_PLUS", 10)
_API_BINS = {cat: sorted({_bin(api, 1024)[0] for api in apis}) for cat, apis in SUSPICIOUS_APIS.items()}


def feature_index(name: str) -> int:
    return _IDX[name]


def category_mask(categories) -> np.ndarray:
    cats = set(categories)
    return np.array([c in cats for c in FEATURE_CATEGORIES])


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def transform(ember: np.ndarray) -> np.ndarray:
    """EMBER v2 vectors [n, 2381] (raw scale) -> model features [n, N_FEATURES]."""
    E = np.atleast_2d(np.asarray(ember, dtype=np.float32))
    if E.shape[1] != EMBER_DIM:
        raise ValueError(f"expected EMBER v2 vectors of width {EMBER_DIM}, got {E.shape[1]}")
    s0 = OFFSETS["section"][0]
    derived = np.zeros((E.shape[0], N_FEATURES - EMBER_DIM), dtype=np.float32)
    # .rsrc hashed bins (value = sign * quantity; abs() undoes the hash sign)
    derived[:, 0] = np.abs(E[:, s0 + 5 + 50 + _RSRC_BIN])
    derived[:, 1] = np.abs(E[:, s0 + 5 + _RSRC_BIN])
    derived[:, 2] = np.abs(E[:, s0 + 5 + 100 + _RSRC_BIN])
    h0 = OFFSETS["header"][0]
    derived[:, 3] = (E[:, h0 + 1 + 40 + _MAGIC_PLUS_BIN] * _MAGIC_PLUS_SIGN > 0).astype(np.float32)
    g0 = OFFSETS["general"][0]
    derived[:, 4] = (E[:, g0 + 7] > 0).astype(np.float32)
    f0 = OFFSETS["imports"][0] + 256
    for j, cat in enumerate(SUSPICIOUS_APIS):
        derived[:, 5 + j] = (E[:, [f0 + b for b in _API_BINS[cat]]] != 0).sum(axis=1)
    return signed_log1p(np.hstack([E, derived])).astype(np.float32)


def check_version(stored_version: str) -> None:
    if stored_version != EXTRACTOR_VERSION:
        raise ValueError(f"feature vectors from extractor {stored_version} cannot be mixed with "
                         f"extractor {EXTRACTOR_VERSION}; re-extract from the raw EMBER vectors / binaries")
