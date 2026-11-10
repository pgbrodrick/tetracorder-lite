"""Portability tests for tetrapy.mixmap: arbitrary ENVI layout, grid, scale and band count."""
from pathlib import Path

import numpy as np
import pytest

from tetrapy import mixmap


def write_cube(tmp, data, wl_nm, interleave="bip", dtype="<i2", scale=10000, hdr_name="scene.hdr"):
    """data (lines, samples, bands) reflectance -> ENVI file in ``interleave``."""
    ax = {"bip": (0, 1, 2), "bil": (0, 2, 1), "bsq": (2, 0, 1)}[interleave]
    raw = np.round(data * scale).astype(dtype) if np.dtype(dtype).kind in "iu" else data.astype(dtype)
    path = tmp / "scene.img"
    np.ascontiguousarray(raw.transpose(ax)).tofile(path)
    code = {"<i2": 2, "<f4": 4, "<u2": 12}[dtype]
    lines = ["ENVI", f"samples = {data.shape[1]}", f"lines = {data.shape[0]}", f"bands = {data.shape[2]}",
             "header offset = 0", "file type = ENVI Standard", f"data type = {code}",
             f"interleave = {interleave}", "byte order = 0", "wavelength units = Nanometers",
             "wavelength = {" + ", ".join(f"{w:.3f}" for w in wl_nm) + "}",
             "fwhm = {" + ", ".join("10.0" for _ in wl_nm) + "}"]
    if np.dtype(dtype).kind in "iu":
        lines.append(f"reflectance scale factor = {scale}")
    (tmp / hdr_name).write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("interleave", ["bip", "bil", "bsq"])
@pytest.mark.parametrize("dtype", ["<i2", "<f4"])
def test_cube_reader_layouts(tmp_path, interleave, dtype):
    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.6, (7, 5, 23))
    wl = np.linspace(400, 2500, 23)
    p = write_cube(tmp_path, data, wl, interleave, dtype)
    cube = mixmap.Cube(p)
    assert (cube.lines, cube.samples, cube.bands) == (7, 5, 23)
    np.testing.assert_allclose(cube.wl, wl / 1000, atol=1e-6)   # header holds 3 decimals (nm)
    refl, bad = cube.block(2, 6)
    np.testing.assert_allclose(refl, data[2:6], atol=1e-4 if dtype == "<i2" else 1e-6)
    assert not bad.any()


def test_cube_nodata_only_counts_usable_channels(tmp_path):
    data = np.full((2, 3, 10), 0.3)
    data[0, 0, 4] = -0.9999            # nodata on a channel that will be deleted
    data[1, 2, 1] = -0.9999            # nodata on a usable channel
    p = write_cube(tmp_path, data, np.linspace(400, 2500, 10))
    (tmp_path / "scene.hdr").write_text((tmp_path / "scene.hdr").read_text() + "data ignore value = -9999\n")
    cube = mixmap.Cube(p)
    valid = np.ones(10, bool); valid[4] = False
    _, bad = cube.block(0, 2, valid)
    assert bad.tolist() == [[False, False, False], [False, False, True]]


def test_deleted_channels_specs():
    wl = np.linspace(0.4, 2.5, 211)
    assert mixmap.parse_channel_spec("1t4 75t79 218 280t285c")[:5] == [1, 2, 3, 4, 75]
    bad = mixmap.deleted_from_ranges(wl, "0.93-0.963, 1.327-1.477")
    assert bad[(wl >= 1.33) & (wl <= 1.47)].all() and not bad[wl < 0.9].any()


def test_spline_weights_interpolate_and_are_natural():
    xk = np.array([0.7, 0.78, 0.84, 0.95])
    y = np.array([0.30, 0.34, 0.36, 0.33])
    np.testing.assert_allclose(y @ mixmap.natural_spline_weights(xk, xk), y, atol=1e-12)
    e = 1e-4
    f = y @ mixmap.natural_spline_weights(xk, np.array([xk[0], xk[0] + e, xk[0] + 2 * e]))
    assert abs((f[2] - 2 * f[1] + f[0]) / e ** 2) < 0.05     # ~0 (natural end); interior values are O(10)


def test_featureless_material_scores_pure_and_mixed():
    """Method A: a pixel that is the library scores F=1, a=1; a 30% mixture scores F~1, a~0.3."""
    wl = np.linspace(2.0, 2.5, 120)
    lib = 0.5 * (1 - 0.3 * np.exp(-0.5 * ((wl - 2.2) / 0.015) ** 2))
    block = {"id": "x", "features": [{"feature_type": "DLw", "continuum": [2.10, 2.13, 2.27, 2.30]}]}
    m = mixmap.FeaturelessMaterial(block, lib, wl, np.ones(wl.size, bool))
    flat = np.full(wl.size, 0.5)
    X = np.stack([lib, 0.3 * lib + 0.7 * flat, flat])
    F, a, ok = m.score(X, gate_depth=0.005)
    assert F[0] == pytest.approx(1, abs=1e-9) and a[0] == pytest.approx(1, abs=1e-9)
    assert F[1] > 0.99 and a[1] == pytest.approx(0.3, abs=0.03)
    assert ok.tolist() == [True, True, False]       # featureless pixel fails the D gate


def test_tetra_material_depth_and_fit():
    """Method B: tetracorder depth of a 30%-deep band at full strength is ~0.30, fit ~1."""
    wl = np.linspace(2.0, 2.5, 120)
    lib = 0.5 * (1 - 0.3 * np.exp(-0.5 * ((wl - 2.2) / 0.015) ** 2))
    block = {"id": "x", "group": 2, "features": [{"feature_type": "DLw", "continuum": [2.10, 2.13, 2.27, 2.30]}]}
    m = mixmap.TetraMaterial(block, lib, wl, np.ones(wl.size, bool))
    F, D = m.features(np.stack([lib, 0.5 * lib + 0.25]))
    assert F[0, 0] == pytest.approx(1, abs=1e-6)
    assert D[0, 0] == pytest.approx(0.3, abs=0.01)
    assert D[0, 1] == pytest.approx(0.15, abs=0.01)  # half-strength mixture -> half depth


def test_library_fixed_slots_for_any_band_count(tmp_path):
    """Convolved libraries keep master record numbers (fixed 6-record slots) for < 256 channels."""
    from tetrapy.conv.library import build_library, FIRST_SPECTRUM
    from tetrapy.conv.specpr import SpecprFile
    master = Path(__file__).parents[1] / "tetracorder/sl1/usgs/library06.conv/sprlb06b"
    if not master.exists():
        pytest.skip("master library not present")
    wl = np.linspace(400, 2450, 150)
    hdr = tmp_path / "g.hdr"
    hdr.write_text("ENVI\nwavelength units = Nanometers\nwavelength = {" + ", ".join(f"{w:.2f}" for w in wl)
                   + "}\nfwhm = {" + ", ".join("12" for _ in wl) + "}\n")
    build_library(master, tmp_path / "lib", tmp_path / "g")
    lib = SpecprFile.open(tmp_path / "lib")
    assert lib.read_spectrum(6).size == 150
    heads = [r for r in lib.spectra() if r >= FIRST_SPECTRUM]
    assert all((r - FIRST_SPECTRUM) % 6 == 0 for r in heads)
