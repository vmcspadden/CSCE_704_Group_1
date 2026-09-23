from pathlib import Path

import numpy as np

from src.features.pe_parser import EMBER_DIM, OFFSETS, ember_vector, raw_features

FIXTURE = Path(__file__).parent / "fixtures" / "distlib_t32.exe"  # benign launcher from distlib


def test_real_pe_parses_with_ember_names():
    raw = raw_features(FIXTURE.read_bytes())
    assert raw["parsed"]
    assert raw["header"]["coff"]["machine"] == "I386"
    assert "CHARA_32BIT_MACHINE" in raw["header"]["coff"]["characteristics"]
    assert raw["header"]["optional"]["magic"] == "PE32"
    assert raw["section"]["entry"] == ".text"
    assert "kernel32.dll" in {k.lower() for k in raw["imports"]}
    vec, ok = ember_vector(FIXTURE.read_bytes())
    assert ok and vec.shape == (EMBER_DIM,) and np.isfinite(vec).all()
    a, _ = OFFSETS["histogram"]
    assert abs(vec[a:a + 256].sum() - 1.0) < 1e-5


def test_entry_name_hashed_per_character_like_ember():
    vec, _ = ember_vector(FIXTURE.read_bytes())
    s = OFFSETS["section"][0] + 5 + 150
    # ".text" -> 5 characters hashed individually ('t' twice), so |sum| of bins is at most 5
    assert np.abs(vec[s:s + 50]).sum() <= 5 and np.count_nonzero(vec[s:s + 50]) >= 3


def test_garbage_and_empty_input_do_not_crash():
    for data in (b"", b"MZ", b"not a pe file" * 100):
        vec, ok = ember_vector(data)
        assert not ok and vec.shape == (EMBER_DIM,)
