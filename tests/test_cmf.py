import numpy as np
import pytest

from tetrapy import cmf


WL = np.linspace(0.4, 2.5, 211)


def gauss(center, width, depth):
    return 1 - depth * np.exp(-0.5 * ((WL - center) / width) ** 2)


def test_feature_window_union_and_types():
    valid = np.ones(WL.size, dtype=bool)
    valid[(WL > 2.15) & (WL < 2.16)] = False
    feats = [
        {"feature_type": "DLw", "continuum": [2.10, 2.12, 2.24, 2.26]},
        {"feature_type": "OLw", "continuum": [2.30, 2.31, 2.38, 2.40]},
        {"feature_type": "DCw", "continuum": [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]},
    ]
    w = cmf.feature_window(feats, WL, valid)
    assert WL[w].min() >= 1.0 and WL[w].max() <= 2.40
    assert ((WL[w] >= 1.0) & (WL[w] <= 1.35)).any()               # curved included by default
    lin = cmf.feature_window(feats, WL, valid, feature_types=cmf.LINEAR_FEATURES)
    assert WL[lin].min() >= 2.10 and WL[lin].max() <= 2.40
    assert not ((WL[w] > 2.15) & (WL[w] < 2.16)).any()
    assert not ((WL[w] > 2.26) & (WL[w] < 2.30)).any()

    only_d = cmf.feature_window(feats, WL, valid, feature_types=("DLw",))
    assert WL[only_d].max() <= 2.26


def test_match_targets_glob():
    ids = ["kaolwxl", "kaolpxl", "muscovite-med-Al", "dickite"]
    assert cmf.match_targets(ids, "kaol*") == [1, 2]
    assert cmf.match_targets(ids, ["dickite", "musc*"]) == [3, 4]
    assert cmf.match_targets(ids, "nope") == []


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(1)
    lib = [gauss(2.20, 0.015, 0.3), gauss(2.33, 0.02, 0.3), gauss(2.165, 0.012, 0.3)]
    feats = [[2.15, 2.16, 2.24, 2.25], [2.27, 2.28, 2.38, 2.39], [2.12, 2.13, 2.19, 2.20]]
    valid = np.ones(WL.size, dtype=bool)
    windows = [cmf.feature_window([{"feature_type": "DLw", "continuum": f}], WL, valid) for f in feats]
    targets = [lib[i][w] for i, w in enumerate(windows)]

    nx, ny = 6, 200
    cube = np.empty((nx, ny, WL.size))
    for x in range(nx):
        a = rng.uniform(0.2, 0.4, ny)[:, None]
        cube[x] = a * (1 + 0.1 * (WL - 1.5)) + rng.normal(0, 0.002, (ny, WL.size))
    for m, ys in enumerate((slice(10, 20), slice(40, 50), slice(70, 80))):
        cube[:, ys] = 0.5 * cube[:, ys] + 0.5 * lib[m] * 0.3
    good = np.ones((nx, ny), dtype=bool)
    good[0, :3] = False
    return cube, good, windows, targets


def test_run_cmf_recovers_implants(scene):
    cube, good, windows, targets = scene
    scores, rank, norm = cmf.run_cmf(cube, good, windows, targets, rank_depth=3, threshold=0.05, threshold_type="raw")

    assert scores.shape == (3, 200, 6) and scores.dtype == np.float32
    assert rank.shape == (3, 200, 6) and rank.dtype == np.uint16
    assert norm.shape == (3, 6)

    for m, ys in enumerate((slice(10, 20), slice(40, 50), slice(70, 80))):
        assert (rank[0, ys] == m + 1).mean() > 0.9

    # Thresholding: everything is either nodata, 0, or >= threshold
    s = scores[scores != -9999]
    assert ((s == 0) | (s >= 0.05)).all()
    # Invalid pixels are nodata and unranked
    assert (scores[:, :3, 0] == -9999).all() and (rank[:, :3, 0] == 0).all()


def test_parallel_matches_serial(scene):
    cube, good, windows, targets = scene
    a = cmf.run_cmf(cube, good, windows, targets, rank_depth=3, cores=1)
    b = cmf.run_cmf(cube, good, windows, targets, rank_depth=3, cores=2)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_identify_single():
    scores = np.zeros((4, 1, 4), dtype=np.float32)
    scores[:, 0, 0] = [0.1, 0.2, 0.3, 0.4]
    scores[:, 0, 1] = [0.5, 0.0, 0.0, 0.0]
    scores[:, 0, 3] = -9999
    rank = np.zeros((3, 1, 4), dtype=np.uint16)
    rank[:, 0, 0] = [4, 3, 2]       # target 2 at position 3
    rank[:, 0, 1] = [1, 0, 0]       # not a target
    rank[:, 0, 2] = [3, 0, 0]       # target 3 at position 1, score 0 here

    s, m, p = cmf.identify_single(scores, rank, [2, 3], top_n=3)
    assert m[0].tolist() == [3, 0, 3, 0]          # best-ranked target wins (3 before 2)
    assert p[0].tolist() == [2, 0, 1, 0]
    assert s[0, 0] == pytest.approx(0.3)
    assert s[0, 1] == 0 and s[0, 3] == -9999

    _, m1, _ = cmf.identify_single(scores, rank, [2], top_n=2)
    assert m1[0, 0] == 0                          # target 2 only at rank 3

    with pytest.raises(ValueError):
        cmf.identify_single(scores, rank, [2], top_n=4)


def ortho_like(scene):
    """Flatten the scene and pad it with -9999 fill, as an orthorectified swath edge."""
    cube, _, windows, targets = scene
    pixels = cube.reshape(-1, cube.shape[-1]).copy()
    fill = np.full((400, pixels.shape[1]), -9999.0)
    pixels = np.vstack([fill[:200], pixels, fill[200:]])
    return pixels, windows, targets


def test_valid_pixels_excludes_nodata():
    x = np.full((4, 5), 0.3)
    x[1, 2] = -9999
    x[2, 0] = np.nan
    x[3, 4] = -9999            # outside the checked channels
    good = cmf.valid_pixels(x, np.array([0, 1, 2]), nodata=-9999)
    assert good.tolist() == [True, False, False, True]


def test_scene_moments_match_numpy(scene):
    pixels, windows, _ = ortho_like(scene)
    used = np.unique(np.concatenate(windows))
    good = cmf.valid_pixels(pixels, used, -9999)
    idx = np.flatnonzero(good)
    mu, S = cmf.scene_moments(pixels, idx, used, chunk_pixels=137)
    ref = pixels[idx][:, used]
    np.testing.assert_allclose(mu, ref.mean(axis=0))
    np.testing.assert_allclose(S, np.cov(ref.T), rtol=1e-8, atol=1e-14)
    assert np.abs(mu).max() < 1          # fill values never reached the statistics


def test_full_scene_recovers_implants_and_ignores_fill(scene):
    pixels, windows, targets = ortho_like(scene)
    used = np.unique(np.concatenate(windows))
    good = cmf.valid_pixels(pixels, used, -9999)
    scores, rank, norm = cmf.run_cmf_full_scene(
        pixels, good, windows, targets, rank_depth=3, threshold=0.05, threshold_type="raw", chunk_pixels=500)

    assert scores.shape == (3, pixels.shape[0]) and rank.shape == (3, pixels.shape[0])
    assert norm.shape == (3,)
    assert (scores[:, ~good] == -9999).all() and (rank[:, ~good] == 0).all()

    # Back to (x, y): implants were placed on lines 10-20, 40-50, 70-80 of every column
    cube = scene[0]
    r0 = rank[0, 200:-200].reshape(cube.shape[:2])
    for m, ys in enumerate((slice(10, 20), slice(40, 50), slice(70, 80))):
        assert (r0[:, ys] == m + 1).mean() > 0.9

    # Same answer whether or not the fill is present: it is excluded, not just masked
    core = pixels[200:-200]
    sc2, rk2, _ = cmf.run_cmf_full_scene(
        core, np.ones(core.shape[0], bool), windows, targets, rank_depth=3, threshold=0.05, threshold_type="raw")
    np.testing.assert_allclose(scores[:, 200:-200], sc2, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(rank[:, 200:-200], rk2)


def test_full_scene_parallel_matches_serial(scene):
    pixels, windows, targets = ortho_like(scene)
    good = cmf.valid_pixels(pixels, np.unique(np.concatenate(windows)), -9999)
    a = cmf.run_cmf_full_scene(pixels, good, windows, targets, rank_depth=3, chunk_pixels=300, cores=1)
    b = cmf.run_cmf_full_scene(pixels, good, windows, targets, rank_depth=3, chunk_pixels=300, cores=2)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_curved_feature_window_spans_all_endpoints():
    valid = np.ones(WL.size, dtype=bool)
    feats = [{"feature_type": "MCw", "continuum": [0.70, 0.72, 0.76, 0.78, 0.83, 0.85, 0.94, 0.97]}]
    assert cmf.feature_window(feats, WL, valid, feature_types=cmf.LINEAR_FEATURES).size == 0
    w = cmf.feature_window(feats, WL, valid)
    assert WL[w].min() >= 0.70 and WL[w].max() <= 0.97
    assert WL[w].max() > 0.93                                        # reaches the last endpoint


def test_match_targets_listed_materials_and_errors():
    import pandas as pd
    table = pd.DataFrame({
        "id": ["ree_nd_ancylite_g21", "ree_samarium_oxide_g21"],
        "title": ["Ancylite_REE WS321", "Samarium_Oxide GDS36"],
        "materials": ["neodymium", "samarium"],
        "n_channels": [0, 21],
        "note": ["only DCw/MCw features, none of cmf.feature_types [...]", ""],
    })
    assert cmf.match_targets(table, "neod*") == [1]          # via list materials
    assert cmf.match_targets(table, "SAMAR*") == [2]         # case-insensitive
    assert cmf.match_targets(table, "*ancylite*") == [1]     # via id
    with pytest.raises(ValueError, match="cmf.feature_types"):
        cmf.resolve_targets({"name": "single", "target": "neod*"}, table)
    with pytest.raises(ValueError, match="Materials: neodymium, samarium"):
        cmf.resolve_targets({"name": "single", "target": "zzz"}, table)
    assert cmf.resolve_targets({"name": "single", "target": "samar*"}, table) == [2]


def test_threshold_types():
    """sigma thresholds on mf * sqrt(s^T C^-1 s); raw thresholds on mf itself."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20000, 4))
    window = np.arange(4)
    targets = [np.array([0.02, 0.0, 0.0, 0.0]), np.array([2.0, 2.0, 0.0, 0.0])]
    f = cmf.window_filters(window, [0, 1], X.mean(0), np.eye(4), targets)
    norm = cmf.filters_norm([f], 2)
    sig = np.sqrt(f[4])                                  # sqrt(s^T C^-1 s) per material

    raw_sc, _ = cmf.score_pixels(X, [f], norm, 2, 2, threshold=0.05, threshold_type="raw")
    sg_sc, _ = cmf.score_pixels(X, [f], norm, 2, 2, threshold=3.0, threshold_type="sigma")
    mf, _ = cmf.score_pixels(X, [f], norm, 2, 2, threshold=-np.inf, threshold_type="raw")

    np.testing.assert_array_equal(raw_sc > 0, mf >= 0.05)
    np.testing.assert_array_equal(sg_sc > 0, mf * sig[:, None] >= 3.0)
    # sigma units are N(0,1) under the background, whatever the target's strength:
    # ~0.13% exceed 3 sigma for both a weak and a strong target
    for m in range(2):
        assert 0.0005 < (sg_sc[m] > 0).mean() < 0.003
    # ...while a raw 0.05 cut is not a false-alarm rate at all: here it passes ~half
    # of pure-background pixels, because 0.05 is a small fraction of these targets
    assert (raw_sc > 0).mean() > 0.4
