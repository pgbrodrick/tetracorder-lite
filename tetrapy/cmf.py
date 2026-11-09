"""
Competitive matched filter (CMF) over a tetracorder mineral group.

For every material in one tetracorder group, this runs a reflectance-mode matched
filter in the style of ``emit_ghg.diffmf`` (``--reflectance_mode``, derivative order
0), restricted to the channels that the expert system's features span for that
material: every ``f`` feature, linear (``DLw``/``MLw``/``OLw``/``WLw``) or curved
(``DCw``/``MCw``/``OCw``/``WCw``). Exclusion features (``n... NOT``: don't map the
material if that feature is present) are not used. The materials then compete per
pixel on a score normalized for their differing window sizes.

Matched filter
--------------
Per cross-track column (pushbroom detector) -- or, with ``full_scene``, once over the
whole image -- and per feature window ``W``::

    mu  = background mean over valid pixels         (on W)
    C   = shrinkage covariance of the background    (on W; Theiler et al. 2012 LOO)
    s   = t_W - mu                                  (library spectrum minus mean)
    mf  = (x_W - mu)^T C^-1 s / (s^T C^-1 s)

which is ``diffmf``'s ``(x - mu) Cstd . (t - mu) Cstd / |(t - mu) Cstd|^2`` written
without forming ``sqrtm(inv(C))``. ``mf`` reads as a linear mixing fraction between
the column background (0) and the library spectrum (1).

Normalization for ranking
-------------------------
Under the background model ``mf`` has standard deviation ``1 / sqrt(s^T C^-1 s)``, so
``mf * sqrt(s^T C^-1 s)`` is a whitened SNR with unit variance whatever the window
size. That SNR still grows with the number of channels a strong feature spans, so the
ranking score divides it by ``sqrt(n_channels)``::

    rank_score = mf * sqrt(s^T C^-1 s / n_channels)

The per-column factor ``sqrt(s^T C^-1 s / n_channels)`` is written to the ``norm``
output, so the ranking score can be rebuilt from the score stack.

Products
--------
``scores`` : float32 GeoTIFF, one band per material in the group, holding the raw
    MF score. Scores that fail ``threshold`` are set to 0; invalid pixels are
    ``nodata``. DEFLATE-compressed. By default (``threshold_type: sigma``) a score
    passes when ``mf * sqrt(s^T C^-1 s) >= threshold``: it sits ``threshold`` background
    standard deviations above the mean. ``threshold_type: raw`` compares ``mf`` itself.
``rank`` : uint16 GeoTIFF, ``rank_depth`` bands. Band ``k`` holds the 1-based stack
    index of the material ranked ``k``-th at that pixel, or 0 when fewer than ``k``
    materials cleared ``threshold``.
``minerals`` : CSV mapping stack index -> tetracorder id / title / library record /
    window channels.
``norm`` : ``.npy`` of the normalization factors, ``(n_minerals, n_cols)`` column-wise
    or ``(n_minerals,)`` with ``full_scene``.

Background
----------
Column-wise (default) suits raw pushbroom geometry, where each image column is one
detector element with its own response. Orthorectified scenes break that mapping, so
``full_scene`` fits a single background from every valid pixel instead. Either way,
pixels that are non-finite, below -0.05, or equal to the reflectance nodata value
(``rfl_nodata``, default -9999) on any window channel are left out of the statistics
and written as ``nodata``.

Modes
-----
``single`` : hunt for one mineral type (an id or glob over tetracorder ids). A pixel
    is identified as that type when any of its top ``top_n`` ranked materials match;
    the best-ranked match is reported with its matched filter score.
"""

import fnmatch
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


Logger = logging.getLogger(__name__)

# Feature types (D=diagnostic, M=must, O=optional, W=weak; L=linear, C=curved
# continuum). All are used by default. A curved feature's window is the full span of
# its 8 endpoints. Exclusion ("NOT") features are never parsed as features, so they
# never contribute channels.
LINEAR_FEATURES = ("DLw", "MLw", "OLw", "WLw")
CURVED_FEATURES = ("DCw", "MCw", "OCw", "WCw")
ALL_FEATURES = LINEAR_FEATURES + CURVED_FEATURES
LIBRARY_KEYS = {"splib06": "reflib", "sprlb06": "reslib"}
SPECPR_DELETED = -1e30          # anything below this is the specpr -1.23e34 sentinel


# ---------------------------------------------------------------------------------
# Expert system / library
# ---------------------------------------------------------------------------------

def find_expert(path: Union[str, Path]) -> Path:
    """
    Resolve the expert system file from a file or a tetracorder ``.cmds`` directory.

    When ``path`` is a directory, the lexically latest ``cmd.lib.setup.t*`` file is
    used (e.g. ``t6.00a6`` over ``t6.00a5``), matching what ``cmd-setup-tetrun`` runs.
    """
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(p for p in path.glob("cmd.lib.setup.t*") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No cmd.lib.setup.t* expert system file under {path}")
    return candidates[-1]


def feature_window(
    features: Sequence[dict],
    wavelengths: np.ndarray,
    valid: np.ndarray,
    feature_types: Iterable[str] = ALL_FEATURES,
) -> np.ndarray:
    """
    Channels spanned by a material's features.

    Each feature contributes every valid channel between its outermost continuum
    endpoints -- ``[a, d]`` for a linear ``(a, b, c, d)`` continuum, the full span of
    the 8 endpoints for a curved one -- covering both continuum windows and the
    absorption between them. The material's window is the union over its features of
    the requested types.

    Parameters
    ----------
    features : sequence of dict
        ``block["features"]`` from :class:`tetrapy.tetracorder.TetraDecoder`.
    wavelengths : numpy.ndarray
        Scene band centres in microns, ``(n_bands,)``.
    valid : numpy.ndarray
        Boolean usable-channel mask, ``(n_bands,)``.
    feature_types : iterable of str
        Feature types to include.

    Returns
    -------
    numpy.ndarray
        Sorted channel indices (0-based), possibly empty.
    """
    feature_types = set(feature_types)
    mask = np.zeros(len(wavelengths), dtype=bool)
    for feat in features or []:
        if feat.get("feature_type") not in feature_types:
            continue
        cont = feat.get("continuum", [])
        try:
            ends = [float(v) for v in cont]
        except (TypeError, ValueError):
            Logger.warning(f"Skipping feature with unresolved continuum {cont!r}")
            continue
        if len(ends) < 4:
            Logger.warning(f"Skipping feature with short continuum {cont!r}")
            continue
        mask |= (wavelengths >= min(ends)) & (wavelengths <= max(ends))
    return np.flatnonzero(mask & valid)


def load_minerals(
    blocks: Sequence[dict],
    libraries: Mapping[str, Union[str, Path]],
    wavelengths: np.ndarray,
    valid: np.ndarray,
    feature_types: Iterable[str] = ALL_FEATURES,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """
    Build the per-material windows and library targets for a group.

    Parameters
    ----------
    blocks : sequence of dict
        Decoded expert system blocks for the group.
    libraries : mapping
        Library id (``splib06`` / ``sprlb06``) -> convolved specpr library path.
    wavelengths, valid : numpy.ndarray
        Scene band centres (microns) and usable-channel mask.
    feature_types : iterable of str
        Feature types defining the windows.

    Returns
    -------
    table : pandas.DataFrame
        One row per block, in block order (stack index = row + 1).
    windows : list of numpy.ndarray
        Channel indices per material (empty when it cannot be evaluated).
    targets : list of numpy.ndarray
        Library reflectance on each material's window.
    """
    from tetrapy.conv import SpecprFile

    opened = {}
    rows, windows, targets = [], [], []
    for i, block in enumerate(blocks, start=1):
        lib_id = block.get("library")
        rec = block.get("record")
        note = ""
        spectrum = None

        if lib_id not in libraries:
            note = f"no library configured for {lib_id!r}"
        else:
            if lib_id not in opened:
                opened[lib_id] = SpecprFile.open(libraries[lib_id])
            try:
                spectrum = opened[lib_id].read_spectrum(rec).astype(np.float64)
            except (IndexError, ValueError) as e:
                note = f"library record {rec} unreadable: {e}"

        window = np.array([], dtype=int)
        if spectrum is not None:
            if spectrum.size != wavelengths.size:
                note = f"library has {spectrum.size} channels, scene has {wavelengths.size}"
            else:
                lib_ok = np.isfinite(spectrum) & (spectrum > SPECPR_DELETED)
                window = feature_window(block.get("features"), wavelengths, valid & lib_ok, feature_types)
                if window.size == 0:
                    have = sorted({f.get("feature_type") for f in block.get("features") or []})
                    skipped = [t for t in have if t not in set(feature_types)]
                    if not have:
                        note = "no features defined"
                    elif skipped and not set(have) & set(feature_types):
                        note = (f"its features are all {'/'.join(skipped)}, which cmf.feature_types "
                                f"excludes (allowed: {list(feature_types)})")
                    else:
                        note = "feature windows fall entirely on deleted channels"

        if note:
            Logger.warning(f"[{i:03}] {block.get('id')}: {note}; scores will be 0")

        windows.append(window)
        targets.append(spectrum[window] if window.size else np.array([]))
        rows.append({
            "index":      i,
            "id":         block.get("id"),
            "title":      block.get("title"),
            "materials":  "; ".join(block.get("materials") or []),
            "library":    lib_id,
            "record":     rec,
            "n_channels": int(window.size),
            "wl_min":     float(wavelengths[window].min()) if window.size else np.nan,
            "wl_max":     float(wavelengths[window].max()) if window.size else np.nan,
            "channels":   " ".join(str(c + 1) for c in window),   # 1-based, like tetracorder
            "note":       note,
        })

    return pd.DataFrame(rows), windows, targets


# ---------------------------------------------------------------------------------
# Covariance (after emit_ghg.diffmf)
# ---------------------------------------------------------------------------------

def _cov(X: np.ndarray) -> np.ndarray:
    return np.cov(X.T, ddof=1)


def fit_looshrinkage_alpha(data: np.ndarray, alphas: np.ndarray) -> float:
    """
    LOO-CV shrinkage intensity (Theiler et al. 2012), diagonal target.

    Vectorized form from ``emit_ghg.diffmf.fit_looshrinkage_alpha``: whiten ``S`` by
    the target once, eigendecompose, and evaluate the LOO negative log-likelihood for
    every alpha in closed form.
    """
    stability_scaling = 100.0
    n, p = data.shape
    alphas = np.asarray(alphas, dtype=np.float64)

    X = (data - data.mean(axis=0)) * stability_scaling
    S = _cov(X)
    Td = np.clip(np.diag(S).copy(), np.finfo(np.float64).tiny, None)
    tinv_sqrt = 1.0 / np.sqrt(Td)
    logdetT = np.sum(np.log(Td))

    Sw = S * tinv_sqrt[:, None] * tinv_sqrt[None, :]
    lam, U = np.linalg.eigh(0.5 * (Sw + Sw.T))
    lam = np.clip(lam, 0.0, None)

    Y2 = ((X * tinv_sqrt) @ U) ** 2

    beta = (1.0 - alphas) / (n - 1.0)
    d = n * beta[:, None] * lam[None, :] + alphas[:, None]
    logdetG = logdetT + np.sum(np.log(d), axis=1)
    R = (1.0 / d) @ Y2.T
    q = 1.0 - beta[:, None] * R

    with np.errstate(invalid="ignore", divide="ignore"):
        nll = 0.5 * (p * np.log(2.0 * np.pi) + logdetG) + \
              (1.0 / (2.0 * n)) * (np.log(q) + R / q).sum(axis=1)
    nll[~np.isfinite(nll)] = np.inf

    m = int(np.argmin(nll))
    return float(alphas[m]) if np.isfinite(nll[m]) else 0.0


DEFAULT_ALPHAS = 10.0 ** np.arange(-10, 0 + 0.05, 0.05)


def shrink(S: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink a covariance toward its diagonal: ``(1 - alpha) S + alpha diag(S)``."""
    return (1.0 - alpha) * S + alpha * np.diag(np.diag(S))


def covariance(data: np.ndarray, style: str = "looshrinkage", fixed_alpha: Optional[float] = None) -> np.ndarray:
    """Covariance of ``data`` (pixels x channels), ``empirical`` or ``looshrinkage`` (as in diffmf)."""
    _check_style(style)
    S = _cov(data)
    if style == "empirical":
        return S
    alpha = fit_looshrinkage_alpha(data, DEFAULT_ALPHAS) if fixed_alpha is None else float(fixed_alpha)
    return shrink(S, alpha)


def _check_style(style: str) -> None:
    if style not in ("looshrinkage", "empirical"):
        raise ValueError(f"Unknown covariance style {style!r}, expected 'looshrinkage' or 'empirical'")


def _inv_spd(C: np.ndarray) -> np.ndarray:
    D, V = np.linalg.eigh(C)
    if D.max() <= 0:
        raise np.linalg.LinAlgError("non-positive covariance")
    D = np.clip(D, D.max() * 1e-12, None)
    return (V / D) @ V.T


# ---------------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------------

def group_windows(windows: Sequence[np.ndarray]) -> Dict[Tuple[int, ...], List[int]]:
    """Map each distinct (non-empty) window to the materials (0-based) that use it."""
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for m, w in enumerate(windows):
        if len(w):
            groups.setdefault(tuple(int(c) for c in w), []).append(m)
    return groups


def valid_pixels(cube: np.ndarray, channels: np.ndarray, nodata: Optional[float] = -9999.0) -> np.ndarray:
    """
    Pixels usable for background statistics and scoring.

    A pixel is valid when, on every channel in ``channels``, it is finite, is not the
    reflectance ``nodata`` value (e.g. the -9999 fill outside an orthorectified swath)
    and exceeds -0.05 (diffmf's sanity floor).

    Parameters
    ----------
    cube : numpy.ndarray
        Reflectance with bands on the last axis, ``(..., n_bands)``.
    channels : numpy.ndarray
        Channel indices to check (the union of the material windows).
    nodata : float, optional
        Reflectance nodata value; ``None`` to skip that check.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``cube.shape[:-1]``.
    """
    good = np.ones(cube.shape[:-1], dtype=bool)
    for c in channels:           # band-by-band keeps peak memory at one band
        band = cube[..., c]
        ok = np.isfinite(band) & (band > -0.05)
        if nodata is not None:
            ok &= ~np.isclose(band, nodata)
        good &= ok
    return good


def window_filters(
    window: np.ndarray,
    members: Sequence[int],
    mu: np.ndarray,
    C: np.ndarray,
    targets: Sequence[np.ndarray],
) -> Optional[tuple]:
    """
    Matched filters for the materials sharing one window.

    Returns ``(window, members, mu, W, snorm)`` with ``W = (t - mu)^T C^-1`` per
    member (``(k, p)``) and ``snorm = s^T C^-1 s`` (``(k,)``, 0 where degenerate), or
    ``None`` when ``C`` cannot be inverted.
    """
    try:
        Cinv = _inv_spd(C)
    except np.linalg.LinAlgError:
        return None
    S = np.stack([targets[m] for m in members]) - mu
    W = S @ Cinv
    snorm = np.einsum("kp,kp->k", W, S)
    snorm = np.where(snorm > 0, snorm, 0.0)
    return np.asarray(window), list(members), mu, W, snorm


def filters_norm(filters: Sequence[tuple], n_minerals: int) -> np.ndarray:
    """Ranking normalization ``sqrt(s^T C^-1 s / n_channels)`` per material."""
    norm = np.zeros(n_minerals, dtype=np.float32)
    for w, members, _, _, snorm in filters:
        norm[members] = np.sqrt(snorm / w.size)
    return norm


def score_pixels(
    X: np.ndarray,
    filters: Sequence[tuple],
    norm: np.ndarray,
    n_minerals: int,
    rank_depth: int,
    threshold: float,
    threshold_type: str = "sigma",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply fitted matched filters to a block of valid pixels and rank them.

    Parameters
    ----------
    X : numpy.ndarray
        Reflectance of valid pixels, ``(n_pixels, n_bands)``.
    filters : sequence of tuple
        From :func:`window_filters`.
    norm : numpy.ndarray
        From :func:`filters_norm`.
    threshold, threshold_type
        ``sigma``: keep a score when ``mf * sqrt(s^T C^-1 s) >= threshold``, i.e. it is
        ``threshold`` background standard deviations above the mean. ``raw``: keep it
        when ``mf >= threshold``.

    Returns
    -------
    scores : float32 ``(n_minerals, n_pixels)``, raw MF; 0 where below ``threshold``
    rank : uint16 ``(rank_depth, n_pixels)``, 1-based stack indices, 0 = none
    """
    n = X.shape[0]
    mf = np.zeros((n, n_minerals), dtype=np.float64)
    for w, members, mu, W, snorm in filters:
        Xw = X[:, w].astype(np.float64, copy=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            mf[:, members] = ((Xw - mu) @ W.T) / snorm
    mf[~np.isfinite(mf)] = 0.0
    if threshold_type == "sigma":
        sig = np.zeros(n_minerals)
        for _, members, _, _, snorm in filters:
            sig[members] = np.sqrt(snorm)
        keep = mf * sig[None, :] >= threshold
    else:
        keep = mf >= threshold

    rank = np.zeros((rank_depth, n), dtype=np.uint16)
    if rank_depth:
        rs = np.where(keep, mf * norm[None, :], -np.inf)
        if rank_depth < n_minerals:
            part = np.argpartition(-rs, rank_depth - 1, axis=1)[:, :rank_depth]
        else:
            part = np.broadcast_to(np.arange(n_minerals), rs.shape)
        order = np.take_along_axis(part, np.argsort(-np.take_along_axis(rs, part, 1), axis=1, kind="stable"), 1)
        order = order[:, :rank_depth]
        ranked = np.take_along_axis(rs, order, 1)
        ids = (order + 1).astype(np.uint16)
        ids[~np.isfinite(ranked)] = 0
        rank[: ids.shape[1]] = ids.T

    mf[~keep] = 0.0
    return mf.T.astype(np.float32), rank


def column_cmf(
    col: np.ndarray,
    good: np.ndarray,
    groups: Mapping[Tuple[int, ...], List[int]],
    targets: Sequence[np.ndarray],
    n_minerals: int,
    rank_depth: int,
    threshold: float,
    style: str = "looshrinkage",
    fixed_alpha: Optional[float] = None,
    min_pixels: int = 10,
    nodata: float = -9999.0,
    threshold_type: str = "sigma",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Competitive matched filter for one cross-track column.

    Parameters
    ----------
    col : numpy.ndarray
        Column reflectance, ``(n_lines, n_bands)``.
    good : numpy.ndarray
        Valid-pixel mask, ``(n_lines,)``. Used for the background statistics and as
        the set of pixels scored.
    groups : mapping
        Output of :func:`group_windows`.
    targets : sequence of numpy.ndarray
        Library reflectance on each material's window.
    n_minerals, rank_depth : int
        Stack size and number of rank bands to keep.
    threshold, threshold_type
        Scores below this are set to 0 and excluded from the ranking; see
        :func:`score_pixels`.
    nodata : float
        Output value for invalid pixels.

    Returns
    -------
    scores : float32 ``(n_minerals, n_lines)``
    rank : uint16 ``(rank_depth, n_lines)``
    norm : float32 ``(n_minerals,)`` -- ``sqrt(s^T C^-1 s / n_channels)``
    """
    n_lines = col.shape[0]
    scores = np.full((n_minerals, n_lines), nodata, dtype=np.float32)
    rank = np.zeros((rank_depth, n_lines), dtype=np.uint16)

    idx = np.flatnonzero(good)
    if idx.size < min_pixels:
        return scores, rank, np.zeros(n_minerals, dtype=np.float32)

    X = col[idx].astype(np.float64)
    filters = []
    for window, members in groups.items():
        Xw = X[:, list(window)]
        f = window_filters(window, members, Xw.mean(axis=0), covariance(Xw, style, fixed_alpha), targets)
        if f is None:
            Logger.debug(f"Singular covariance on window {window[0]}-{window[-1]}; skipping {len(members)} materials")
        else:
            filters.append(f)

    norm = filters_norm(filters, n_minerals)
    scores[:, idx], rank[:, idx] = score_pixels(
        X, filters, norm, n_minerals, rank_depth, threshold, threshold_type)
    return scores, rank, norm


# Shared state for forked workers (avoids pickling the cube per task)
_STATE: dict = {}


def _single_thread_blas() -> None:
    """
    Pin BLAS to one thread in a worker process.

    Each worker otherwise inherits a full BLAS thread pool, and ``cores`` processes x
    ``n_cpu`` threads oversubscribes the machine badly (several times slower than
    serial). Uses ``threadpoolctl`` when installed, else calls OpenBLAS's setter on
    the libraries mapped into this (Linux) process.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
        return
    except ImportError:
        pass

    import ctypes
    try:
        with open("/proc/self/maps") as f:
            libs = {line.split()[-1] for line in f if "openblas" in line.lower() and "/" in line}
    except OSError:
        return
    for lib in libs:
        try:
            handle = ctypes.CDLL(lib)
        except OSError:
            continue
        for sym in ("openblas_set_num_threads", "openblas_set_num_threads64_",
                    "scipy_openblas_set_num_threads", "scipy_openblas_set_num_threads64_"):
            if hasattr(handle, sym):
                getattr(handle, sym)(1)


def _map(func, tasks: Sequence, collect, cores: int, label: str) -> None:
    """Run ``func`` over ``tasks`` (forked pool when ``cores > 1``), feeding ``collect``."""
    done, step = 0, max(1, len(tasks) // 20)

    def tick():
        nonlocal done
        done += 1
        if done % step == 0:
            Logger.info(f"CMF: {done}/{len(tasks)} {label}")

    if cores and cores > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(cores, initializer=_single_thread_blas) as pool:
            for res in pool.imap_unordered(func, tasks):
                collect(res)
                tick()
    else:
        if cores and cores > 1:
            Logger.warning("fork start method unavailable; running CMF serially")
        for t in tasks:
            collect(func(t))
            tick()


def _work_columns(cols: Sequence[int]) -> List[tuple]:
    s = _STATE
    return [(x, *column_cmf(s["cube"][x], s["good"][x], **s["kwargs"])) for x in cols]


def run_cmf(
    cube: np.ndarray,
    good: np.ndarray,
    windows: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    rank_depth: int = 10,
    threshold: float = 5.0,
    threshold_type: str = "sigma",
    style: str = "looshrinkage",
    fixed_alpha: Optional[float] = None,
    min_pixels: int = 10,
    nodata: float = -9999.0,
    cores: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise competitive matched filter (raw, unorthorectified scenes).

    Parameters
    ----------
    cube : numpy.ndarray
        Reflectance as ``(n_cols, n_lines, n_bands)`` -- cross-track first, so each
        ``cube[x]`` is one detector column.
    good : numpy.ndarray
        Valid-pixel mask, ``(n_cols, n_lines)``.
    windows, targets : sequence of numpy.ndarray
        From :func:`load_minerals`.
    cores : int
        Worker processes (fork). ``1`` runs serially.

    Returns
    -------
    scores : float32 ``(n_minerals, n_lines, n_cols)``
    rank : uint16 ``(rank_depth, n_lines, n_cols)``
    norm : float32 ``(n_minerals, n_cols)``
    """
    _check_style(style)
    n_cols, n_lines, _ = cube.shape
    n_min = len(windows)
    rank_depth = int(min(rank_depth or n_min, n_min))
    groups = group_windows(windows)
    Logger.info(f"CMF (column-wise): {n_min} materials over {len(groups)} distinct windows, "
                f"{n_cols} columns x {n_lines} lines, rank depth {rank_depth}")

    scores = np.empty((n_min, n_lines, n_cols), dtype=np.float32)
    rank = np.empty((rank_depth, n_lines, n_cols), dtype=np.uint16)
    norm = np.empty((n_min, n_cols), dtype=np.float32)

    def collect(results):
        for x, sc, rk, nm in results:
            scores[:, :, x] = sc
            rank[:, :, x] = rk
            norm[:, x] = nm

    _STATE.update(cube=cube, good=good, kwargs=dict(
        groups=groups, targets=targets, n_minerals=n_min, rank_depth=rank_depth,
        threshold=threshold, threshold_type=threshold_type, style=style,
        fixed_alpha=fixed_alpha, min_pixels=min_pixels, nodata=nodata))
    try:
        chunks = [list(range(i, min(i + 8, n_cols))) for i in range(0, n_cols, 8)]
        _map(_work_columns, chunks, collect, cores, "column blocks")
    finally:
        _STATE.clear()

    return scores, rank, norm


def scene_moments(
    pixels: np.ndarray,
    idx: np.ndarray,
    channels: np.ndarray,
    chunk_pixels: int = 65536,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and (ddof=1) covariance over ``channels`` of the pixels ``pixels[idx]``.

    Two passes over pixel chunks in float64, so the full-scene covariance never needs
    a float64 copy of the scene. Every window's empirical covariance is a submatrix
    of this one.
    """
    n = idx.size
    total = np.zeros(channels.size)
    for i in range(0, n, chunk_pixels):
        total += pixels[idx[i:i + chunk_pixels]][:, channels].sum(axis=0, dtype=np.float64)
    mu = total / n

    S = np.zeros((channels.size, channels.size))
    for i in range(0, n, chunk_pixels):
        D = pixels[idx[i:i + chunk_pixels]][:, channels].astype(np.float64) - mu
        S += D.T @ D
    return mu, S / (n - 1)


def _work_alpha(task: Tuple[int, np.ndarray]) -> Tuple[int, float]:
    k, wi = task
    return k, fit_looshrinkage_alpha(_STATE["sample"][:, wi], DEFAULT_ALPHAS)


def _work_pixels(bounds: Tuple[int, int]) -> tuple:
    s = _STATE
    lo, hi = bounds
    X = s["pixels"][s["idx"][lo:hi]]
    return (lo, hi, *score_pixels(X, s["filters"], s["norm"], s["n_minerals"], s["rank_depth"], s["threshold"],
                                    s["threshold_type"]))


def run_cmf_full_scene(
    pixels: np.ndarray,
    good: np.ndarray,
    windows: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    rank_depth: int = 10,
    threshold: float = 5.0,
    threshold_type: str = "sigma",
    style: str = "looshrinkage",
    fixed_alpha: Optional[float] = None,
    min_pixels: int = 10,
    nodata: float = -9999.0,
    cores: int = 1,
    alpha_pixels: int = 50000,
    chunk_pixels: int = 65536,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-scene competitive matched filter (orthorectified scenes).

    One background (mean and covariance per window) is estimated from every valid
    pixel in the scene, instead of one per detector column; orthorectification mixes
    detector columns, so the column-wise model no longer applies. Invalid pixels
    (including the reflectance nodata fill) are excluded from the statistics and
    written as ``nodata``.

    Parameters
    ----------
    pixels : numpy.ndarray
        Reflectance as ``(n_pixels, n_bands)`` (the scene flattened).
    good : numpy.ndarray
        Valid-pixel mask, ``(n_pixels,)``.
    windows, targets : sequence of numpy.ndarray
        From :func:`load_minerals`.
    alpha_pixels : int
        For ``looshrinkage``, the shrinkage intensity is fit by LOO-CV on a random
        subsample of this many valid pixels (the LOO cost scales with pixels x
        alphas). The covariance itself always uses every valid pixel. Ignored when
        ``fixed_alpha`` is set or ``style`` is ``empirical``.
    chunk_pixels : int
        Pixels per scoring task (bounds the per-worker memory).
    seed : int
        Seed for the alpha subsample.

    Returns
    -------
    scores : float32 ``(n_minerals, n_pixels)``
    rank : uint16 ``(rank_depth, n_pixels)``
    norm : float32 ``(n_minerals,)``
    """
    _check_style(style)
    n_pix = pixels.shape[0]
    n_min = len(windows)
    rank_depth = int(min(rank_depth or n_min, n_min))
    groups = group_windows(windows)

    scores = np.full((n_min, n_pix), nodata, dtype=np.float32)
    rank = np.zeros((rank_depth, n_pix), dtype=np.uint16)

    idx = np.flatnonzero(good)
    Logger.info(f"CMF (full scene): {n_min} materials over {len(groups)} distinct windows, "
                f"{idx.size}/{n_pix} valid pixels, rank depth {rank_depth}")
    if idx.size < max(min_pixels, 2):
        Logger.warning(f"Only {idx.size} valid pixels; nothing scored")
        return scores, rank, np.zeros(n_min, dtype=np.float32)

    used = np.unique(np.concatenate([np.asarray(w) for w in groups]))
    pos = {int(c): i for i, c in enumerate(used)}
    mu_all, S_all = scene_moments(pixels, idx, used, chunk_pixels)

    sample = None
    if style == "looshrinkage" and fixed_alpha is None:
        rng = np.random.default_rng(seed)
        take = min(int(alpha_pixels), idx.size)
        sample = pixels[np.sort(rng.choice(idx, take, replace=False))][:, used].astype(np.float64)
        Logger.info(f"Fitting shrinkage on a {take}-pixel subsample")

    items = list(groups.items())
    wis = [np.array([pos[c] for c in window]) for window, _ in items]

    alphas = np.zeros(len(items))
    if style == "looshrinkage":
        if sample is None:
            alphas[:] = float(fixed_alpha)
        else:
            def put(res):
                alphas[res[0]] = res[1]
            _STATE.update(sample=sample)
            try:
                _map(_work_alpha, list(enumerate(wis)), put, cores, "shrinkage fits")
            finally:
                _STATE.clear()
            del sample
        Logger.debug(f"Shrinkage alpha: median {np.median(alphas):.3g}, max {np.max(alphas):.3g}")

    filters = []
    for (window, members), wi, alpha in zip(items, wis, alphas):
        S = S_all[np.ix_(wi, wi)]
        C = S if style == "empirical" else shrink(S, alpha)
        f = window_filters(window, members, mu_all[wi], C, targets)
        if f is None:
            Logger.warning(f"Singular covariance on window {window[0]}-{window[-1]}; "
                           f"{len(members)} materials will score 0")
        else:
            filters.append(f)

    norm = filters_norm(filters, n_min)

    def collect(res):
        lo, hi, sc, rk = res
        scores[:, idx[lo:hi]] = sc
        rank[:, idx[lo:hi]] = rk

    _STATE.update(pixels=pixels, idx=idx, filters=filters, norm=norm, n_minerals=n_min,
                  rank_depth=rank_depth, threshold=threshold, threshold_type=threshold_type)
    try:
        tasks = [(i, min(i + chunk_pixels, idx.size)) for i in range(0, idx.size, chunk_pixels)]
        _map(_work_pixels, tasks, collect, cores, "pixel blocks")
    finally:
        _STATE.clear()

    return scores, rank, norm


# ---------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------

def match_targets(table: Union[pd.DataFrame, Sequence[str]], target: Union[str, Sequence[str]]) -> List[int]:
    """
    1-based stack indices of the materials matching any glob in ``target``.

    Each glob is matched case-insensitively against the tetracorder ``id``, the
    ``title``, and each name in the block's ``list materials`` (``materials``), so
    ``neod*`` finds every ``ree_nd_*`` entry listing ``neodymium``, and ``kaol*``
    finds kaolinite entries by id.

    Parameters
    ----------
    table : pandas.DataFrame or sequence of str
        The minerals table (``id`` and optionally ``title`` / ``materials`` columns),
        or just the list of ids.
    target : str or sequence of str
        Glob(s).
    """
    if isinstance(table, pd.DataFrame):
        cols = [c for c in ("id", "title", "materials") if c in table]
        rows = table[cols].fillna("").astype(str).to_dict("records")
    else:
        rows = [{"id": str(t)} for t in table]

    patterns = [target] if isinstance(target, str) else list(target)
    patterns = [str(p).lower() for p in patterns]

    hits = []
    for i, row in enumerate(rows, start=1):
        names = [row.get("id", ""), row.get("title", "")]
        names += [m.strip() for m in row.get("materials", "").split(";") if m.strip()]
        if any(fnmatch.fnmatchcase(n.lower(), p) for n in names if n for p in patterns):
            hits.append(i)
    return hits


def resolve_targets(mode: Optional[Mapping], table: pd.DataFrame) -> List[int]:
    """Validate ``mode.target`` against the group up front, with a helpful error."""
    if not mode or mode.get("name") != "single":
        return []
    target = mode.get("target")
    if not target:
        raise ValueError("cmf.mode.target is required for mode 'single'")
    idx = match_targets(table, target)
    if not idx:
        mats = sorted({m.strip() for v in table.get("materials", pd.Series(dtype=str)).fillna("")
                       for m in str(v).split(";") if m.strip()})
        raise ValueError(
            f"cmf.mode.target {target!r} matches no id, title or listed material in this group. "
            f"Ids: {', '.join(table['id'].astype(str))}"
            + (f". Materials: {', '.join(mats)}" if mats else ""))
    usable = [i for i in idx if table["n_channels"].iloc[i - 1] > 0] if "n_channels" in table else idx
    if not usable:
        notes = sorted({str(table["note"].iloc[i - 1]) for i in idx}) if "note" in table else []
        hint = ""
        if any("feature_types" in n for n in notes):
            hint = (" Their feature types are excluded by cmf.feature_types; remove that "
                    "override to use all of them.")
        raise ValueError(f"cmf.mode.target {target!r} matches {len(idx)} materials, none with a "
                         f"usable window, so none could ever be identified ({'; '.join(notes)}).{hint}")
    return idx


def identify_single(
    scores: np.ndarray,
    rank: np.ndarray,
    target_idx: Sequence[int],
    top_n: int,
    nodata: float = -9999.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-mineral identification.

    A pixel is identified when any of its top ``top_n`` ranked materials is one of
    ``target_idx``; the best-ranked such material is reported.

    Returns
    -------
    score : float32 ``(n_lines, n_cols)``
        Matched filter score of the identified material (0 when not identified,
        ``nodata`` where the stack is nodata).
    mineral : uint16 ``(n_lines, n_cols)``
        1-based stack index of the identified material, 0 when not identified.
    position : uint16 ``(n_lines, n_cols)``
        1-based rank position at which it was found, 0 when not identified.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if top_n > rank.shape[0]:
        raise ValueError(f"top_n={top_n} exceeds the {rank.shape[0]} rank bands computed; "
                         f"raise cmf.rank_depth")
    if not len(target_idx):
        raise ValueError("No target materials to identify")

    top = rank[:top_n]
    hit = np.isin(top, np.asarray(target_idx, dtype=top.dtype))
    found = hit.any(axis=0)
    first = hit.argmax(axis=0)

    mineral = np.take_along_axis(top, first[None], 0)[0].astype(np.uint16)
    mineral[~found] = 0
    position = np.where(found, first + 1, 0).astype(np.uint16)

    gather = np.clip(mineral.astype(np.int64) - 1, 0, None)
    score = np.take_along_axis(scores, gather[None], 0)[0].astype(np.float32)
    score[~found] = 0.0
    score[scores[0] == nodata] = nodata
    return score, mineral, position


# ---------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------

def _write_tif(data: np.ndarray, like, path: Union[str, Path], names: Sequence[str], nodata) -> None:
    import xarray as xr
    import rioxarray  # noqa: F401  (registers .rio)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": np.arange(1, data.shape[0] + 1), "y": like["y"], "x": like["x"]},
        attrs={"long_name": tuple(str(n) for n in names)},
    )
    # Transform is derived from the x/y coords; carry the CRS when the scene has one
    if like.rio.crs is not None:
        da = da.rio.write_crs(like.rio.crs)
    if nodata is not None:
        da = da.rio.write_nodata(nodata)

    predictor = 3 if np.issubdtype(data.dtype, np.floating) else 2
    Logger.info(f"Saving {path}")
    da.rio.to_raster(path, driver="GTiff", compress="DEFLATE", predictor=predictor,
                     tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")


def _read_tif(path: Union[str, Path]) -> np.ndarray:
    import xarray as xr
    with xr.open_dataset(path, engine="rasterio", mask_and_scale=False) as ds:
        return ds["band_data"].load().data


def build(
    rfl: str,
    expert: Union[str, Path],
    group: int,
    reflib: str,
    reslib: str,
    output: Mapping,
    mode: Optional[Mapping] = None,
    deleted_channels=None,
    feature_types: Sequence[str] = ALL_FEATURES,
    covariance_style: str = "looshrinkage",
    fixed_alpha: Optional[float] = None,
    threshold: float = 5.0,
    threshold_type: str = "sigma",
    rank_depth: int = 10,
    min_pixels: int = 10,
    nodata: float = -9999.0,
    rfl_nodata: Optional[float] = -9999.0,
    full_scene: bool = False,
    alpha_pixels: int = 50000,
    cores: int = 1,
    reuse: bool = False,
) -> None:
    """
    Run the competitive matched filter for one tetracorder group, then the mode.

    \b
    Parameters
    ----------
    rfl : str
        Scene reflectance raster (data file; its ``.hdr`` supplies the grid).
    expert : str or Path
        Expert system file, or the tetracorder ``.cmds`` directory holding it.
    group : int
        Tetracorder group whose materials compete (e.g. 1 or 2).
    reflib, reslib : str
        Convolved ``splib06`` / ``sprlb06`` specpr libraries for this scene.
    output : mapping
        ``scores``, ``rank``, ``minerals`` (CSV) and ``norm`` (``.npy``) paths.
    mode : mapping, optional
        ``name`` (``single``) plus mode options; for ``single``: ``target`` (id or
        glob, or a list of them), ``top_n``, ``output``.
    deleted_channels : str or mapping, optional
        ``sensor.deleted_channels`` spec; these channels are never used.
    feature_types : sequence of str
        Feature types defining the windows (default: all linear and curved types;
        exclusion ``NOT`` features are never used).
    covariance_style, fixed_alpha
        Covariance estimate, as diffmf's ``--covariance_style`` / ``--fixed_alpha``.
    threshold : float
        Scores below this are written as 0 and excluded from the ranking.
    threshold_type : str
        ``sigma`` (default): ``threshold`` is in background standard deviations,
        ``mf * sqrt(s^T C^-1 s)``, so it means the same thing for every material and
        scene. ``raw``: ``threshold`` is on the raw MF score (mixing fraction), which
        for weak absorbers like the REE is often far below 0.05 even at high SNR.
    rank_depth : int
        Number of rank bands written (0 or None for all materials).
    min_pixels : int
        Minimum good pixels in a column (or the scene) to estimate a background.
    nodata : float
        Output nodata value for the score stack / single-mode product.
    rfl_nodata : float, optional
        Reflectance nodata value (default -9999). Pixels carrying it on any window
        channel are excluded from the background statistics and written as ``nodata``.
        ``None`` disables the check (non-finite values are always excluded).
    full_scene : bool
        Estimate one background over the whole image instead of per detector column.
        Required for orthorectified (pre-ortho'd) scenes, where image columns are no
        longer detector columns.
    alpha_pixels : int
        Full-scene only: pixels subsampled to fit the LOO shrinkage alpha.
    cores : int
        Worker processes.
    reuse : bool
        When the scores/rank/minerals outputs already exist, load them and only run
        the mode.
    """
    import xarray as xr
    from tetrapy.conv.library import read_grid
    from tetrapy.sensor import resolve_deleted_channels
    from tetrapy.tetracorder import TetraDecoder

    out = {k: Path(v) for k, v in (output or {}).items() if v}
    with xr.open_dataset(rfl, engine="rasterio") as ds:
        src = ds["band_data"]
        like = src.isel(band=0, drop=True)
        nbands = src.sizes["band"]

        if reuse and all(k in out and out[k].exists() for k in ("scores", "rank", "minerals")):
            Logger.info("Reusing existing CMF products")
            scores = _read_tif(out["scores"])
            rank = _read_tif(out["rank"]).astype(np.uint16)
            table = pd.read_csv(out["minerals"])
        else:
            expert = find_expert(expert)
            Logger.info(f"Decoding group {group} from {expert}")
            blocks = TetraDecoder(expert, groups=(group,), raise_casts=False).blocks
            if not blocks:
                raise ValueError(f"Group {group} has no materials in {expert}")

            grid = read_grid(rfl)
            if grid.nbands != nbands:
                raise ValueError(f"Header has {grid.nbands} wavelengths but raster has {nbands} bands")

            valid = np.ones(nbands, dtype=bool)
            if deleted_channels:
                chans = resolve_deleted_channels(deleted_channels, rfl, nbands)
                valid[np.asarray(chans, dtype=int) - 1] = False
                Logger.info(f"Excluding {len(chans)} deleted channels")

            table, windows, targets = load_minerals(
                blocks, {"splib06": reflib, "sprlb06": reslib}, grid.wavelengths, valid, feature_types)

            used = np.unique(np.concatenate([w for w in windows if len(w)] or [np.array([], int)]))
            if used.size == 0:
                raise ValueError(f"No group {group} material has a usable window "
                                 f"(feature_types={list(feature_types)})")
            resolve_targets(mode, table)      # fail before the (long) filter run, not after

            Logger.info("Loading reflectance")
            data = src.load().data                                              # (band, y, x)
            n_lines, n_cols = data.shape[1:]
            if threshold_type not in ("sigma", "raw"):
                raise ValueError(f"cmf.threshold_type must be 'sigma' or 'raw', got {threshold_type!r}")
            common = dict(rank_depth=rank_depth or len(windows), threshold=threshold,
                          threshold_type=threshold_type,
                          style=covariance_style, fixed_alpha=fixed_alpha,
                          min_pixels=min_pixels, nodata=nodata, cores=cores)

            if full_scene:
                pixels = np.ascontiguousarray(data.transpose(1, 2, 0)).reshape(-1, nbands)
                del data
                good = valid_pixels(pixels, used, rfl_nodata)
                Logger.info(f"Excluding {good.size - good.sum()} invalid / nodata pixels")
                scores, rank, norm = run_cmf_full_scene(
                    pixels, good, windows, targets, alpha_pixels=alpha_pixels, **common)
                del pixels
                scores = scores.reshape(-1, n_lines, n_cols)
                rank = rank.reshape(-1, n_lines, n_cols)
            else:
                cube = np.ascontiguousarray(data.transpose(2, 1, 0))           # (x, y, band)
                del data
                good = valid_pixels(cube, used, rfl_nodata)
                Logger.info(f"Excluding {good.size - good.sum()} invalid / nodata pixels")
                scores, rank, norm = run_cmf(cube, good, windows, targets, **common)
                del cube

            write_products(out, scores, rank, norm, table, like, nodata)
            frac = (rank[0] > 0).mean() if rank.shape[0] else 0
            Logger.info(f"CMF complete: {frac:.1%} of pixels have a material above threshold "
                        f"{threshold}{'σ' if threshold_type == 'sigma' else ' (raw MF)'}")

        if mode:
            run_mode(mode, scores, rank, table, like, nodata)


def write_products(out: Mapping[str, Path], scores, rank, norm, table: pd.DataFrame, like, nodata) -> None:
    """Write whichever of the scores / rank / minerals / norm outputs are configured."""
    if "scores" in out:
        _write_tif(scores, like, out["scores"], [f"{r.index}: {r.id}" for r in table.itertuples()], nodata)
    if "rank" in out:
        _write_tif(rank, like, out["rank"], [f"rank {k}" for k in range(1, rank.shape[0] + 1)], 0)
    if "minerals" in out:
        out["minerals"].parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out["minerals"], index=False)
    if "norm" in out:
        out["norm"].parent.mkdir(parents=True, exist_ok=True)
        np.save(out["norm"], norm)


def run_mode(mode: Mapping, scores: np.ndarray, rank: np.ndarray, table: pd.DataFrame, like, nodata: float) -> None:
    """Dispatch to the configured identification mode."""
    name = mode.get("name")
    if name in (None, "", "none"):
        return
    if name != "single":
        raise ValueError(f"Unknown cmf.mode.name {name!r}; available: 'single'")

    target_idx = resolve_targets(mode, table)
    Logger.info("Single-mineral targets: " + ", ".join(
        f"{i}:{table['id'].iloc[i - 1]}" for i in target_idx))

    score, mineral, position = identify_single(scores, rank, target_idx, int(mode.get("top_n", 1)), nodata)
    Logger.info(f"Identified {(mineral > 0).mean():.2%} of pixels within the top {mode.get('top_n', 1)}")

    if mode.get("output"):
        stack = np.stack([score, mineral.astype(np.float32), position.astype(np.float32)])
        _write_tif(stack, like, mode["output"], ["mf_score", "mineral_index", "rank_position"], nodata)
