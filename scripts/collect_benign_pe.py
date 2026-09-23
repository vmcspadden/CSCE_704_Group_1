"""Collect a small out-of-distribution BENIGN PE corpus from PyPI Windows wheels.

These binaries are modern (unlike SOREL's 2017-2019 samples), come from
well-known open-source projects, and are never used for training or
threshold calibration - only as an extra false-positive sanity check of the
served model (scripts/test_service.py). Output: data/raw/benign_pypi/<package>__<file>
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402

PACKAGES = ["numpy", "scipy", "pandas", "pywin32", "pillow", "lxml", "cryptography", "psutil", "pyzmq",
            "matplotlib", "scikit-learn", "grpcio", "pyyaml", "cffi", "markupsafe", "kiwisolver", "regex",
            "pycryptodome", "wxpython", "pygame", "orjson", "ujson", "bcrypt", "greenlet", "gevent",
            "pymupdf", "pyinstaller", "setuptools", "pip", "comtypes", "pythonnet", "pywinpty", "debugpy",
            "llvmlite", "shapely", "netcdf4", "h5py", "tables", "pyqt5-qt5", "pyside6-essentials"]
PE_SUFFIXES = (".exe", ".dll", ".pyd", ".sys")


def windows_wheel_url(pkg: str) -> str | None:
    r = requests.get(f"https://pypi.org/pypi/{pkg}/json", timeout=60)
    if r.status_code != 200:
        return None
    files = r.json()["urls"]
    for tag in ("win_amd64", "win32"):
        for f in files:
            if f["filename"].endswith(".whl") and tag in f["filename"]:
                return f["url"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-size", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--max-per-package", type=int, default=40)
    ap.add_argument("--max-wheel-mb", type=int, default=150)
    args = ap.parse_args()
    out = load_config().path("raw_dir") / "benign_pypi"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for pkg in PACKAGES:
        url = windows_wheel_url(pkg)
        if not url:
            continue
        head = requests.head(url, timeout=60, allow_redirects=True)
        if int(head.headers.get("Content-Length", 0)) > args.max_wheel_mb * 1024 * 1024:
            continue
        zf = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=300).content))
        n = 0
        for info in zf.infolist():
            if not info.filename.lower().endswith(PE_SUFFIXES) or info.file_size > args.max_size:
                continue
            data = zf.read(info)
            if data[:2] != b"MZ":
                continue
            (out / f"{pkg}__{Path(info.filename).name}").write_bytes(data)
            n += 1
            if n >= args.max_per_package:
                break
        total += n
        print(f"{pkg}: {n} PE files", flush=True)
    print(f"total {total} files in {out}")


if __name__ == "__main__":
    main()
