"""
Dominant + secondary material mapping from the tetracorder expert system (portable runner).

Implements the two prototypes from ``mf_test/figures/methods.md`` for any imaging
spectrometer whose reflectance is an ENVI cube with a ``wavelength`` (and ``fwhm``) header:

- **Method A, featureless baseline**: per material, continuum-remove every feature with the
  expert system's windows, take absorption vectors ``y = 1 - CR(pixel)``,
  ``s = 1 - CR(library)``, gate on diagnostic (D/M) features, and score
  ``F = sign(y.s)(y.s)^2/((s.s)(y.y))`` (uncentred R^2 of y on s) and ``a = y.s/s.s``.
- **Method B, tetracorder-style fit**: tetracorder's own per-feature fit/depth (tp1mat,
  bandmp, bandmpcv), feature and material constraints, NOT features and T/P limits, keeping
  every material's score instead of only the winner.
- **Two-mineral test** on each method's rank-2/3 candidates (secondary materials).
- **Band-depth check** for a named feature (default: Nd, 0.74 and 0.80 um), as an independent
  reference for weak targets.

Nothing here is tied to EMIT: the channel grid, the deleted channels and the library spectra
all come from the input cube. Libraries are convolved to the cube's grid if not supplied.

Example
-------
::

    python -m tetrapy.mixmap scene_rfl --out out/ --groups 1 2 21 \\
        --deleted-um "0.38-0.405 0.93-0.963 1.11-1.164 1.327-1.477 1.770-1.968 2.455-2.50" \\
        --cores 8
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tetrapy.conv.specpr import DELETED, SpecprFile
    from tetrapy.tetracorder import TetraDecoder
except ImportError:
    # Standalone use (``python tetrapy/mixmap.py``) without the full tetrapy install: the package
    # __init__ needs click/rich/rasterio, but this module only needs numpy (+ pandas for the
    # decoder). Register a bare package so the submodules import directly.
    import types as _types
    _pkg = _types.ModuleType("tetrapy"); _pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules.setdefault("tetrapy", _pkg)
    from tetrapy.conv.specpr import DELETED, SpecprFile
    from tetrapy.tetracorder import TetraDecoder


Logger = logging.getLogger("tetrapy.mixmap")

# Defaults that reproduce the EMIT 2023-02-20 analysis
DEFAULTS = dict(
    fit_min_a=0.5,          # Method A: rank-1 needs F >= this to be mapped
    gate_depth=0.005,       # Method A: each D/M feature needs fitted depth >= this
    mix_dr=0.4,             # two-mineral test: residual reduction
    mix_depth=0.01,         # two-mineral test: candidate's fitted depth
    mix_twin=0.95,          # two-mineral test: skip pairs whose library shapes correlate >= this
    temperature_c=(-20.0, 80.0),
    pressure_bar=(0.5, 1.5),
)
# Nd 0.74 / 0.80 um bands: (left continuum, band centre, right continuum) windows in um
ND_BANDS = {
    "nd_0.74": ((0.700, 0.724), (0.738, 0.752), (0.765, 0.785)),
    "nd_0.80": ((0.765, 0.785), (0.795, 0.810), (0.832, 0.853)),
}


# =====================================================================================
# I/O
# =====================================================================================

ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


def read_envi_header(path: Path) -> dict:
    """Parse an ENVI .hdr into a dict (lowercase keys, ``{...}`` lists split on commas)."""
    text = path.read_text(errors="replace")
    out = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{.*?\}|[^\n]*)", text, re.M | re.S):
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val.startswith("{"):
            val = [v.strip() for v in val[1:-1].replace("\n", " ").split(",") if v.strip()]
        out[key] = val
    return out


def header_path(data: Path) -> Path:
    for cand in (Path(str(data) + ".hdr"), data.with_suffix(".hdr")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No ENVI header for {data}")


class Cube:
    """Read-only ENVI cube; ``block(y0, y1)`` returns float32 (lines, samples, bands) reflectance."""

    def __init__(self, data: Path, scale: Optional[float] = None, nodata: Optional[float] = None):
        self.path = Path(data)
        self.hdr_path = header_path(self.path)
        h = read_envi_header(self.hdr_path)
        self.hdr = h
        self.lines, self.samples, self.bands = int(h["lines"]), int(h["samples"]), int(h["bands"])
        dt = np.dtype(ENVI_DTYPES[int(h.get("data type", 4))])
        self.dtype = dt.newbyteorder(">" if int(h.get("byte order", 0)) == 1 else "<")
        self.interleave = str(h.get("interleave", "bsq")).lower()
        self.offset = int(h.get("header offset", 0))
        shape = {"bil": (self.lines, self.bands, self.samples),
                 "bip": (self.lines, self.samples, self.bands),
                 "bsq": (self.bands, self.lines, self.samples)}[self.interleave]
        self.mm = np.memmap(self.path, dtype=self.dtype, mode="r", offset=self.offset, shape=shape)

        wl = np.array([float(v) for v in h["wavelength"]])
        units = str(h.get("wavelength units", "")).lower()
        to_um = 1e-3 if (units.startswith(("nan", "nm")) or (not units and wl.max() > 100)) else 1.0
        self.wl = wl * to_um
        self.fwhm = np.array([float(v) for v in h["fwhm"]]) * to_um if "fwhm" in h else None
        if not np.all(np.diff(self.wl) > 0):
            raise ValueError("wavelengths must be strictly increasing")

        # reflectance scale: explicit > header 'reflectance scale factor' > guess from dtype
        if scale is None:
            if "reflectance scale factor" in h:
                scale = 1.0 / float(h["reflectance scale factor"])
            elif dt.kind in "iu":
                scale = 1e-4
                Logger.warning("Integer cube with no 'reflectance scale factor'; assuming 1e-4 "
                               "(10000 = 1.0). Override with --scale.")
            else:
                scale = 1.0
        self.scale = float(scale)
        if nodata is None:
            nodata = float(h["data ignore value"]) if "data ignore value" in h else -9999.0
        self.nodata = nodata

        self.bbl = None
        if "bbl" in h:
            self.bbl = np.array([float(v) for v in h["bbl"]]) > 0

    def block(self, y0: int, y1: int, valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Lines y0:y1 -> (reflectance float32 (n, samples, bands), bad-pixel mask (n, samples)).

        A pixel is bad if any *usable* channel is nodata, non-finite, or below -0.05
        reflectance (deleted channels are ignored, since some products fill them)."""
        if self.interleave == "bil":
            x = np.asarray(self.mm[y0:y1]).transpose(0, 2, 1)
        elif self.interleave == "bip":
            x = np.asarray(self.mm[y0:y1])
        else:
            x = np.asarray(self.mm[:, y0:y1]).transpose(1, 2, 0)
        raw = x.astype(np.float32)
        v = slice(None) if valid is None else valid
        bad = np.zeros(raw.shape[:2], bool)
        if self.nodata is not None:
            bad |= np.isclose(raw[..., v], self.nodata).any(-1)
        refl = raw * np.float32(self.scale)
        with np.errstate(invalid="ignore"):
            bad |= ~np.isfinite(refl[..., v]).all(-1) | (refl[..., v] < -0.05).any(-1)
        return refl, bad


def write_envi(path: Path, data: np.ndarray, names: Sequence[str], like: Cube, dtype, nodata=None,
               description: str = ""):
    """Write a (bands, lines, samples) BSQ ENVI file, carrying map info from ``like``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(data.astype(dtype))
    arr.tofile(path)
    code = {v: k for k, v in ENVI_DTYPES.items()}[np.dtype(dtype).str[1:]]
    h = ["ENVI", f"description = {{{description}}}", f"samples = {arr.shape[2]}", f"lines = {arr.shape[1]}",
         f"bands = {arr.shape[0]}", "header offset = 0", "file type = ENVI Standard",
         f"data type = {code}", "interleave = bsq", "byte order = 0",
         "band names = {" + ", ".join(names) + "}"]
    if nodata is not None:
        h.append(f"data ignore value = {nodata}")
    for key in ("map info", "coordinate system string"):
        if key in like.hdr:
            v = like.hdr[key]
            h.append(f"{key} = {{{', '.join(v) if isinstance(v, list) else v}}}")
    Path(str(path) + ".hdr").write_text("\n".join(h) + "\n")


# =====================================================================================
# Channels and libraries
# =====================================================================================

def parse_channel_spec(spec: str) -> List[int]:
    """Tetracorder channel spec ("1t4 75t79 218c") -> sorted 1-based channels."""
    spec = spec.split("#", 1)[0].strip().rstrip("c").strip()
    out = set()
    for tok in spec.split():
        tok = tok.rstrip("c")
        if not tok:
            continue
        a, _, b = tok.partition("t")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def deleted_from_ranges(wl: np.ndarray, spec: str) -> np.ndarray:
    """'lo-hi lo-hi ...' (um) -> boolean deleted mask."""
    bad = np.zeros(wl.size, bool)
    for tok in spec.replace(",", " ").split():
        lo, hi = (float(v) for v in tok.split("-"))
        bad |= (wl >= lo) & (wl <= hi)
    return bad


def resolve_valid(cube: Cube, args) -> Tuple[np.ndarray, str]:
    """Usable channels: start from all, then remove header bbl, run DELETPTS, CLI channels / ranges."""
    bad = np.zeros(cube.bands, bool)
    src = []
    if cube.bbl is not None:
        bad |= ~cube.bbl
        src.append("header bbl")
    if args.tetracorder_run:
        f = next(Path(args.tetracorder_run).glob("cmds.start.*"), None)
        if f:
            m = re.search(r"^==\[DELETPTS\](.*)$", f.read_text(errors="replace"), re.M)
            if m:
                ch = np.array(parse_channel_spec(m.group(1)))
                bad[ch[ch <= cube.bands] - 1] = True
                src.append(f"[DELETPTS] in {f.name}")
    if args.deleted_channels:
        ch = np.array(parse_channel_spec(args.deleted_channels))
        bad[ch[ch <= cube.bands] - 1] = True
        src.append("--deleted-channels")
    if args.deleted_um:
        bad |= deleted_from_ranges(cube.wl, args.deleted_um)
        src.append("--deleted-um")
    if not src:
        Logger.warning("No deleted channels given (no bbl, no --deleted-um/--deleted-channels, "
                       "no --tetracorder-run). Water bands will be used; results will suffer.")
    return ~bad, ", ".join(src) or "none"


def load_libraries(cube: Cube, args, out: Path) -> Dict[str, SpecprFile]:
    """Convolved splib06/sprlb06 on the cube grid: given, or built from the master libraries."""
    libs = {}
    for key, given, master in (("splib06", args.reflib, "splib06b"), ("sprlb06", args.reslib, "sprlb06b")):
        if given:
            lib = SpecprFile.open(given)
        else:
            try:
                from tetrapy.conv.library import build_library
            except ImportError:          # tetrapy.conv/__init__ is numpy-only, but be explicit
                raise
            src = Path(args.tetracorder_root) / "sl1/usgs/library06.conv" / master
            dst = out / "libs" / key
            if not dst.exists():
                Logger.info(f"Convolving {master} to the cube grid -> {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if cube.fwhm is None:
                    raise ValueError("cube header has no 'fwhm'; supply --reflib/--reslib convolved to this grid")
                grid = dst.parent / "grid"                     # read_grid only reads grid.hdr
                import shutil
                shutil.copyfile(cube.hdr_path, str(grid) + ".hdr")
                build_library(src, dst, grid)
            lib = SpecprFile.open(dst)
        w = lib.read_spectrum(6)
        if w.size != cube.bands or np.abs(w - cube.wl).max() > 2e-4:
            raise ValueError(f"{key} library grid ({w.size} ch) does not match the cube "
                             f"({cube.bands} ch); convolve it to this cube or omit --{'reflib' if key == 'splib06' else 'reslib'}")
        libs[key] = lib
    return libs


# =====================================================================================
# Shared helpers
# =====================================================================================

def family(block: dict) -> frozenset:
    """Mineral family = names in 'list materials', ignoring any '_suffix'."""
    return frozenset(x.split("_")[0].lower() for x in (block.get("materials") or []))


def natural_spline_weights(xk: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """(K, Q) matrix W with natural-cubic-spline(knots xk, values y)(xq) = y @ W."""
    K = len(xk)
    W = np.zeros((K, len(xq)))
    h = np.diff(xk)
    for k in range(K):
        y = np.zeros(K); y[k] = 1.0
        A = np.zeros((K, K)); r = np.zeros(K); A[0, 0] = A[-1, -1] = 1
        for i in range(1, K - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            r[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        M = np.linalg.solve(A, r)
        j = np.clip(np.searchsorted(xk, xq) - 1, 0, K - 2)
        t0, t1 = xk[j], xk[j + 1]; hh = t1 - t0
        W[k] = ((M[j] * (t1 - xq) ** 3 + M[j + 1] * (xq - t0) ** 3) / (6 * hh)
                + (y[j] - M[j] * hh ** 2 / 6) * (t1 - xq) / hh
                + (y[j + 1] - M[j + 1] * hh ** 2 / 6) * (xq - t0) / hh)
    return W


def ramp(x, lo, hi):
    """tetracorder fuzzy gate: 0 below lo, linear to 1 at hi."""
    if hi is None or hi <= lo:
        return (x >= lo).astype(float)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _pair(v):
    v = [float(x) for x in v]
    return (v[0], v[1] if len(v) > 1 else None)


# =====================================================================================
# Method A: featureless baseline
# =====================================================================================

class FeaturelessMaterial:
    """Continuum operators + library absorption for one material (Method A geometry)."""

    def __init__(self, block: dict, lib: np.ndarray, wl: np.ndarray, valid: np.ndarray):
        self.id = block["id"]
        self.ops, self.diag = [], []
        for ft in block["features"]:
            e = ft["continuum"]
            idx = np.flatnonzero((wl >= min(e)) & (wl <= max(e)) & valid)
            if idx.size == 0:
                continue
            wins = []
            for a, b in zip(e[0::2], e[1::2]):
                i = np.flatnonzero((wl >= a) & (wl <= b) & valid)
                if i.size == 0:  # window narrower than a channel, or deleted: nearest USABLE channel
                    vi = np.flatnonzero(valid)
                    i = vi[[np.argmin(abs(wl[vi] - (a + b) / 2))]]
                wins.append(i)
            xs = np.array([wl[i].mean() for i in wins])
            K = len(wins); Wt = np.zeros((K, idx.size))
            for j, x in enumerate(wl[idx]):
                k = int(np.clip(np.searchsorted(xs, x) - 1, 0, K - 2)); t = (x - xs[k]) / (xs[k + 1] - xs[k])
                Wt[k, j] += 1 - t; Wt[k + 1, j] += t
            self.ops.append((idx, wins, Wt))
            self.diag.append(ft["feature_type"][0] in "DM")
        self.ok = bool(self.ops) and not (lib[np.concatenate([o[0] for o in self.ops])] < -1).any()
        self.s = [1 - p for p in self.cr_parts(lib[None])] if self.ok else []
        self.s = [v[0] for v in self.s]

    def cr_parts(self, X):
        out = []
        with np.errstate(all="ignore"):        # zero continua (bad pixels) give inf/nan, scored as 0
            for idx, wins, Wt in self.ops:
                means = np.stack([X[..., i].mean(-1) for i in wins], -1)
                out.append(np.nan_to_num(X[..., idx] / (means @ Wt), nan=1.0, posinf=1.0, neginf=1.0))
        return out

    def score(self, X: np.ndarray, gate_depth: float):
        """X (n, bands) -> fit F, abundance a, eligible mask."""
        n = X.shape[0]
        if not self.ok:
            return np.zeros(n), np.zeros(n), np.zeros(n, bool)
        num = np.zeros(n); sn = 0.0; yy = np.zeros(n); ok = np.ones(n, bool)
        for Yc, s, diag in zip(self.cr_parts(X), self.s, self.diag):
            Y = 1 - Yc
            nf = Y @ s
            num += nf; sn += s @ s; yy += (Y * Y).sum(1)
            if diag:
                af = nf / (s @ s)
                ok &= (af > 0) & (af * s.max() >= gate_depth)
        with np.errstate(all="ignore"):
            ab = num / sn
            fit = np.sign(num) * num ** 2 / np.maximum(sn * yy, 1e-30)
        fit = np.nan_to_num(fit); ab = np.nan_to_num(ab)
        return fit, ab, ok & (ab > 0)


# =====================================================================================
# Method B: tetracorder-style fit
# =====================================================================================

def wtochbin(wl, w1, w2):
    """specpr wtochbin: window (um) -> 0-based inclusive channel range or None."""
    i = np.flatnonzero(wl >= w1)
    if i.size == 0:
        return None
    a = b = int(i[0])
    while b + 1 < wl.size and wl[b + 1] <= w2:
        b += 1
    if a == b and not (w1 <= wl[a] <= w2):
        if a > 0 and abs(wl[a] - w1) > abs(wl[a - 1] - w1):
            a = b = a - 1
        elif a < wl.size - 1 and abs(wl[a] - w1) > abs(wl[a + 1] - w1):
            a = b = a + 1
    return a, b


class TetraMaterial:
    """One material scored as tetracorder does (tp1mat/bandmp/bandmpcv), all pixels at once."""

    def __init__(self, block: dict, lib: np.ndarray, wl: np.ndarray, valid: np.ndarray):
        self.b = block; self.id = block["id"]; self.group = block["group"]
        self.lib = lib
        V = valid & (lib > -1)
        self.feats = []; self.must_disabled = False
        for f in block["features"]:
            e = f["continuum"]; imp = f["feature_type"][0]
            wins = []
            for a, c in zip(e[0::2], e[1::2]):
                r = wtochbin(wl, a, c)
                i = np.arange(r[0], r[1] + 1) if r else np.array([], int)
                wins.append(i[V[i]])
            if any(w.size == 0 for w in wins):          # window entirely deleted -> feature disabled
                self.feats.append(None)
                if imp == "M":
                    self.must_disabled = True
                continue
            fw = wins[1:3] if len(wins) == 4 else wins     # curved: fit inner two windows + band
            span = np.flatnonzero((wl >= wl[fw[0][0]]) & (wl <= wl[fw[-1][-1]]) & V)
            if span.size < 3:
                self.feats.append(None); continue
            xs = np.array([wl[w].mean() for w in wins])
            if len(wins) == 2:
                t = (wl[span] - xs[0]) / (xs[1] - xs[0]); Wt = np.stack([1 - t, t])
            else:
                Wt = natural_spline_weights(xs, wl[span])
            libc = lib[span] / (np.array([lib[w].mean() for w in wins]) @ Wt)
            core = np.flatnonzero((wl[span] > wl[fw[0][-1]]) & (wl[span] < wl[fw[-1][0]]))
            if core.size == 0:
                self.feats.append(None); continue
            cmin = core[np.argmin(libc[core])]; cmax = core[np.argmax(libc[core])]
            ftype = -1 if (libc[cmax] - 1) > (1 - libc[cmin]) else 1   # emission vs absorption
            imin = int(cmax if ftype == -1 else cmin)
            area = float(np.sum(np.abs(1 - libc[core])))

            def lim(key):
                v = f.get(key)
                return (float(v[0]), float(v[1]) if len(v) > 1 else np.inf) if v else (-np.inf, np.inf)

            def pr(key):
                v = f.get(key); return _pair(v) if v else None
            weight = float(f["weight"][0]) if f.get("weight") else 1.0
            self.feats.append(dict(wins=wins, span=span, Wt=Wt, libc=libc, imin=imin, ftype=ftype,
                                   area=max(area, 0) * weight, imp=imp,
                                   ct=lim("ct"), lct=lim("lct"), rct=lim("rct"),
                                   lgr=pr("lct/rct>"), rgl=pr("rct/lct>"),
                                   rbblcgt=pr("rcbblc>"), rbblclt=pr("rcbblc<"),
                                   lbbrcgt=pr("lcbbrc>"), lbbrclt=pr("lcbbrc<"), rbd=pr("r*bd>")))
        nonweak = [f for f in self.feats if f and f["imp"] != "W" and f["area"] > 0]
        tot = sum(f["area"] for f in nonweak) or 1
        for f in self.feats:
            if f:
                f["dln"] = f["area"] / tot if f["imp"] != "W" else 0.0
        self.enabled = bool(nonweak) and not self.must_disabled
        cc = block.get("constituent_constraints", {})
        self.cons = {k: _pair(v) for k, v in cc.items() if v and not isinstance(v[0], str)}
        self.nots = block.get("not_features", [])

    def features(self, X):
        n = X.shape[0]; F = np.zeros((len(self.feats), n)); D = np.zeros_like(F)
        for k, f in enumerate(self.feats):
            if f is None:
                continue
            cm = np.stack([X[:, w].mean(1) for w in f["wins"]], 1)
            cont = cm @ f["Wt"]
            with np.errstate(all="ignore"):
                obs = X[:, f["span"]] / np.where(np.abs(cont) > 1e-20, cont, np.nan)
                L = f["libc"]; dL = L - L.mean(); dO = obs - obs.mean(1, keepdims=True)
                cov = (dO * dL).sum(1); b = cov / (dL ** 2).sum()
                xk = (1 - b) / b; dep = 1 - (L[f["imin"]] + xk) / (xk + 1)
                fit = np.sqrt(np.abs(b * cov / (dO ** 2).sum(1)))
            conref = cont[:, f["imin"]]; cl = cm[:, 0]; cr_ = cm[:, -1]
            bad = ~np.isfinite(dep) | ~np.isfinite(fit) | (fit <= 0) | (fit > 1.1) | (np.abs(dep) < 1e-7)
            for lm, v in ((f["ct"], conref), (f["lct"], cl), (f["rct"], cr_)):
                bad |= (v < lm[0]) | (v > lm[1])
            dep = np.where(bad, 0, dep); fit = np.where(bad, 0, fit)
            g = np.ones(n)
            with np.errstate(all="ignore"):
                safe = lambda x: np.where(np.abs(x) < 1e-7, 1e-7, x)
                if f["lgr"]: g *= ramp(np.minimum(cl, 1e20) / np.maximum(cr_, 1e-7), *f["lgr"])
                if f["rgl"]: g *= ramp(np.minimum(cr_, 1e20) / np.maximum(cl, 1e-7), *f["rgl"])
                bot = conref * (1 - dep)
                if f["rbblcgt"]: g *= ramp((cr_ - bot) / safe(cl - bot), *f["rbblcgt"])
                if f["lbbrcgt"]: g *= ramp((cl - bot) / safe(cr_ - bot), *f["lbbrcgt"])
                if f["rbblclt"]: g *= 1 - ramp((cr_ - bot) / safe(cl - bot), *f["rbblclt"])
                if f["lbbrclt"]: g *= 1 - ramp((cl - bot) / safe(cr_ - bot), *f["lbbrclt"])
                if f["rbd"] is not None:
                    g *= ramp(np.where(dep * f["ftype"] > 1e-6, conref * dep, 0), *f["rbd"])
            g = np.nan_to_num(g)
            F[k] = fit * g; D[k] = dep * g
        return F, D

    def stored(self, F, D):
        """Feature values as tetracorder leaves them for NOT tests (wrong-sign zeroing rules)."""
        F = F.copy(); D = D.copy(); dead = np.zeros(F.shape[1], bool)
        for k, f in enumerate(self.feats):
            if f is None:
                continue
            neg = (D[k] * f["ftype"] <= 1e-6) & ~dead
            if f["imp"] in "DM":
                F[k:, neg] = 0; D[k:, neg] = 0; dead |= neg
            if f["imp"] in "WM":
                F[:, neg] = 0; D[:, neg] = 0; dead |= neg
        return F, D

    def score(self, F, D):
        n = F.shape[1]; kill = np.zeros(n, bool)
        sf = np.zeros(n); sd = np.zeros(n); sfd = np.zeros(n)
        for k, f in enumerate(self.feats):
            if f is None:
                continue
            neg = D[k] * f["ftype"] <= 1e-6
            if f["imp"] in "DMW":
                kill |= neg
            if f["imp"] != "W":
                xx = np.where(neg, 0, 1.0)
                sf += F[k] * xx * f["dln"]; sd += D[k] * f["dln"] * f["ftype"]; sfd += D[k] * f["dln"] * f["ftype"] * F[k]
        for arr in (sf, sd, sfd):
            arr[kill] = 0
        c = self.cons
        for key, val in (("FITALL", sf), ("DEPTHALL", np.abs(sd)), ("FDALL", sfd)):
            if key in c:
                g = ramp(val, *c[key]); sf, sd, sfd = sf * g, sd * g, sfd * g
        fitv, depv = sf.copy(), sd.copy()
        if "FIT" in c: fitv *= ramp(sf, *c["FIT"])
        if "DEPTH" in c: depv *= ramp(np.abs(sd), *c["DEPTH"])
        if "DEPTH-FIT" in c: depv *= ramp(sf, *c["DEPTH-FIT"])
        return fitv, depv


def gtp_ok(lines: List[str], t_c, p_bar) -> bool:
    """Material disabled if its temperature/pressure range (t1..t4) misses the scene's."""
    tk = (t_c[0] + 273.15, t_c[1] + 273.15)
    for l in lines:
        s = l.split("\\#")[0]
        if "constraint:" not in s:
            continue
        m = re.search(r"temperature:\s*([CK])\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", s)
        if m:
            off = 273.15 if m.group(1) == "C" else 0.0
            t1 = max(float(m.group(2)) + off, 0); t4 = float(m.group(5)) + off
            if tk[1] < t1 or tk[0] > t4:
                return False
        m = re.search(r"pressure:\s*bars?\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", s)
        if m and (p_bar[1] < float(m.group(1)) or p_bar[0] > float(m.group(4))):
            return False
    return True


# =====================================================================================
# Group setup
# =====================================================================================

class Group:
    """Everything needed to score one tetracorder group on one instrument grid."""

    def __init__(self, g: int, dec: TetraDecoder, libs, wl, valid, args):
        self.g = g
        order = [dec.parse_block(x) for x in TetraDecoder.extract_blocks(dec.lines)]
        raw = {}
        for x in TetraDecoder.extract_blocks(dec.lines):
            for l in x:
                if l.startswith("ID="):
                    raw[l[3:].strip()] = x; break
        pos = {b["id"]: i for i, b in enumerate(order)}
        nog0 = []
        for l in dec.lines:
            if l.startswith("nogroup0:"):
                nog0 = [int(x) for x in l.split("\\#")[0].split()[1:]]
        self.blocks_a = [b for b in order if b["group"] == g]
        if not self.blocks_a:
            raise ValueError(f"group {g} has no materials")
        self.blocks_b = self.blocks_a + ([b for b in order if b["group"] == 0] if g not in nog0 else [])
        lib_of = lambda b: libs[b["library"]].read_spectrum(b["record"]).astype(float)
        # Method A
        self.A = [FeaturelessMaterial(b, lib_of(b), wl, valid) for b in self.blocks_a]
        # Method B (+ NOT references, resolved like getnotfeat: last earlier entry with that record)
        self.B = [TetraMaterial(b, lib_of(b), wl, valid) for b in self.blocks_b]
        for m in self.B:
            if not gtp_ok(raw.get(m.id, []), args.temperature_c, args.pressure_bar):
                m.enabled = False
        self.refs = {}
        for m in self.B:
            for nf in m.nots:
                prev = [b for b in order[:pos[m.id]] if (b["library"], b["record"]) == (nf["library"], nf["record"])]
                if not prev:
                    Logger.warning(f"{m.id}: NOT reference {nf['ref']} not found; ignored")
                    nf["ref_id"] = None; continue
                nf["ref_id"] = prev[-1]["id"]
                if nf["ref_id"] not in self.refs:
                    self.refs[nf["ref_id"]] = TetraMaterial(prev[-1], lib_of(prev[-1]), wl, valid)
        self.ids_a = [b["id"] for b in self.blocks_a]
        self.ids_b = [b["id"] for b in self.blocks_b]
        self.fam_a = [family(b) for b in self.blocks_a]
        self.fam_b = [family(b) for b in self.blocks_b]
        # two-mineral test geometry (Method-A style continuum removal) for every B material
        self.mixgeo = {b["id"]: FeaturelessMaterial(b, lib_of(b), wl, valid) for b in self.blocks_b}
        self.libsp = {b["id"]: lib_of(b) for b in self.blocks_b}
        self.dmflags = {mid: geo.diag for mid, geo in self.mixgeo.items()}
        nA = sum(m.ok for m in self.A); nB = sum(m.enabled for m in self.B)
        Logger.info(f"group {g}: {len(self.A)} materials (A usable {nA}); B {len(self.B)} incl. group 0 "
                    f"(enabled {nB}), {sum(len(m.nots) for m in self.B)} NOT features")


# =====================================================================================
# Per-block processing (worker)
# =====================================================================================

_S: dict = {}


def _init_worker(state):
    _S.update(state)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _two_mineral(G: Group, ids, fam, dom, cand, X, p):
    """Return (secondary id index 1-based, a2, dRSS) arrays (2, n) for ranks 2-3."""
    n = X.shape[0]
    out = np.zeros((2, n), np.uint16); a2o = np.zeros((2, n), np.float32); dro = np.zeros((2, n), np.float32)
    cache = {}
    for k in range(2):
        c = cand[k]; m = (c > 0) & (dom > 0) & (c != dom)
        if not m.any():
            continue
        idx = np.flatnonzero(m); pv = dom[idx].astype(np.int64) * 10000 + c[idx]
        for v in np.unique(pv):
            k1, k2 = divmod(int(v), 10000)
            if fam[k1 - 1] & fam[k2 - 1]:
                continue
            id1, id2 = ids[k1 - 1], ids[k2 - 1]
            g1, g2 = G.mixgeo[id1], G.mixgeo[id2]
            if not (g1.ok and g2.ok):
                continue
            if (k1, k2) not in cache:
                ops = g2.ops + g1.ops

                def crs(spec, ops=ops):          # bind this pair's windows now
                    out_ = []
                    with np.errstate(all="ignore"):
                        for i_, w_, W_ in ops:
                            means = np.stack([spec[..., i].mean(-1) for i in w_], -1)
                            out_.append(np.nan_to_num(spec[..., i_] / (means @ W_), nan=1.0, posinf=1.0, neginf=1.0))
                    return np.concatenate(out_, -1)
                s1 = 1 - crs(G.libsp[id1][None])[0]; s2 = 1 - crs(G.libsp[id2][None])[0]
                n2 = sum(o[0].size for o in g2.ops)
                with np.errstate(all="ignore"):
                    tw_ = np.corrcoef(s1, s2)[0, 1]
                cache[(k1, k2)] = (crs, s1, s2, n2, tw_ if np.isfinite(tw_) else 1.0)
            crs, s1, s2, n2, tw = cache[(k1, k2)]
            if tw >= p["mix_twin"]:
                continue
            sel = idx[pv == v]
            Y = 1 - crs(X[sel].astype(float))
            S = np.stack([s1, s2], 1)
            A = (Y @ S) @ np.linalg.inv(S.T @ S + 1e-12 * np.eye(2))
            a1s = (Y @ s1) / (s1 @ s1)
            r1 = ((Y - np.outer(a1s, s1)) ** 2).sum(1); r12 = ((Y - A @ S.T) ** 2).sum(1)
            dr = (r1 - r12) / np.maximum(r1, 1e-30)
            ok = (A[:, 0] > 0) & (A[:, 1] > 0) & (dr >= p["mix_dr"]) & (A[:, 1] * s2[:n2].max() >= p["mix_depth"])
            off = 0; res = Y - np.outer(A[:, 0], s1)
            for op, flag in zip(g2.ops, g2.diag):
                q = op[0].size
                if flag:
                    ok &= (res[:, off:off + q] @ s2[off:off + q]) > 0
                off += q
            out[k, sel[ok]] = k2; a2o[k, sel[ok]] = A[ok, 1]; dro[k, sel[ok]] = dr[ok]
    return out, a2o, dro


def _band_depth(X, wl, valid, bands):
    res = []
    for _, (L, C, R) in bands.items():
        def mean_in(lo, hi):
            i = np.flatnonzero((wl >= lo) & (wl <= hi) & valid)
            if i.size == 0:
                i = np.array([np.argmin(abs(wl - (lo + hi) / 2))])
            return X[:, i].mean(1), wl[i].mean()
        (l, xl), (c, xc), (r, xr) = mean_in(*L), mean_in(*C), mean_in(*R)
        with np.errstate(all="ignore"):
            res.append(np.nan_to_num(1 - c / (l + (r - l) * (xc - xl) / (xr - xl))))
    return np.stack(res)


def _process(y0y1):
    y0, y1 = y0y1
    cube, groups, p = _S["cube"], _S["groups"], _S["p"]
    refl, bad = cube.block(y0, y1, _S["valid"])
    ns = refl.shape[1]
    X = refl.reshape(-1, cube.bands).astype(np.float64)
    good = ~bad.ravel()
    Xg = X[good]
    out = {"y0": y0, "y1": y1}
    for g, G in groups.items():
        n = len(Xg)
        # ---- Method A
        FA = np.full((len(G.A), n), -np.inf); AA = np.zeros((len(G.A), n))
        for i, m in enumerate(G.A):
            f, a, ok = m.score(Xg, p["gate_depth"])
            FA[i] = np.where(ok, f, -np.inf); AA[i] = a
        oA = np.argsort(-FA, 0)[:3]; fA = np.take_along_axis(FA, oA, 0)
        rA = np.where(np.isfinite(fA) & (fA > 0), oA + 1, 0).astype(np.uint16)
        domA = np.where(fA[0] >= p["fit_min_a"], rA[0], 0).astype(np.uint16)
        abA = np.take_along_axis(AA, oA, 0)
        # ---- Method B
        refF = {k: r.stored(*r.features(Xg)) for k, r in G.refs.items()}
        FB = np.zeros((len(G.B), n)); DB = np.zeros_like(FB)
        for i, m in enumerate(G.B):
            if not m.enabled:
                continue
            Fm, Dm = m.features(Xg); f, dd = m.score(Fm, Dm)
            if m.nots:
                hit = np.zeros(n, bool)
                for nf in m.nots:
                    if not nf.get("ref_id"):
                        continue
                    rF, rD = refF[nf["ref_id"]]; k = nf["feature"] - 1
                    dep = np.abs(rD[k])
                    if nf["relative_to"]:
                        dep = dep / np.maximum(np.abs(Dm[nf["relative_to"] - 1]), 1e-13)
                    hit |= (dep > nf["depth"]) & (rF[k] > nf["fit"])
                hit &= f > 0
                f = np.where(hit, 0, f); dd = np.where(hit, 0, dd)
            FB[i] = f; DB[i] = dd
        oB = np.argsort(-FB, 0, kind="stable")[:3]; fB = np.take_along_axis(FB, oB, 0)
        rB = np.where(fB > 0, oB + 1, 0).astype(np.uint16)
        bdB = np.take_along_axis(DB, oB[:1], 0)[0]
        domB = np.where((fB[0] > 0) & (bdB > 0), rB[0], 0).astype(np.uint16)
        # ---- secondaries
        secA, a2A, _ = _two_mineral(G, G.ids_a, G.fam_a, domA, rA[1:3], Xg, p)
        secB, a2B, _ = _two_mineral(G, G.ids_b, G.fam_b, domB, rB[1:3], Xg, p)

        def full(a, fill=0, dtype=None):
            a = np.asarray(a)
            shp = a.shape[:-1] + (good.size,)
            o = np.full(shp, fill, dtype or a.dtype); o[..., good] = a
            return o.reshape(a.shape[:-1] + (y1 - y0, ns))
        out[g] = dict(
            A_rank=full(rA), A_fit=full(np.where(np.isfinite(fA), fA, 0).astype(np.float32)),
            A_abund=full(abA.astype(np.float32)), A_dom=full(domA),
            A_sec=full(secA), A_sec_abund=full(a2A),
            B_rank=full(rB), B_fit=full(fB.astype(np.float32)), B_depth=full(bdB[None].astype(np.float32))[0],
            B_dom=full(domB), B_sec=full(secB), B_sec_abund=full(a2B),
        )
    if p["band_depth"]:
        out["band_depth"] = np.zeros((len(p["band_depth"]), y1 - y0, ns), np.float32)
        bd = _band_depth(Xg, cube.wl, _S["valid"], p["band_depth"])
        tmp = np.zeros((bd.shape[0], good.size), np.float32); tmp[:, good] = bd
        out["band_depth"] = tmp.reshape(-1, y1 - y0, ns)
    out["valid"] = good.reshape(y1 - y0, ns)
    return out


# =====================================================================================
# Optional comparison with a tetracorder run on the same cube
# =====================================================================================

def compare_with_tetracorder(run: Path, g: int, dec: TetraDecoder, prod: dict, G: Group, shape, rows, bd=None, nd_ids=None):
    """Score both methods against a tetracorder run's winner rasters (``rows`` = processed line range)."""
    import gzip
    order = {b["id"]: b for b in dec.blocks + dec.ignored}
    groups_dirs = dec.groups
    gdir = run / groups_dirs.get(g, "")
    ny, nx = shape

    def read(p):
        b = gzip.open(p).read()
        return np.frombuffer(b[nx:], np.uint8)[: ny * nx].reshape(ny, nx)
    truth = np.zeros((ny, nx), np.int32)
    for i, mid in enumerate(G.ids_b):
        b = order[mid]
        p = gdir / (b["filename"] + ".depth.gz")
        if p.exists():
            v = read(p); truth[v > 0] = i + 1
    truth = truth[rows[0]:rows[1]]
    res = {"tc_mapped": float((truth > 0).mean())}
    B = prod["B_dom"]; m = (truth > 0) | (B > 0)
    res["B_agree_exact"] = float((B == truth)[m].mean()) if m.any() else None
    res["B_tc_in_top3"] = float(((prod["B_rank"] == truth[None]) & (truth[None] > 0)).any(0)[truth > 0].mean()) if (truth > 0).any() else None
    # A vs tetracorder (group materials only; family and exact)
    tg = np.where(truth <= len(G.ids_a), truth, 0)                # B order starts with the group's own materials
    A = prod["A_dom"]; both = (tg > 0) & (A > 0)
    if both.any():
        yy, xx = np.nonzero(both)
        s = np.random.default_rng(0).choice(yy.size, min(yy.size, 100000), replace=False)
        res["A_agree_exact"] = float(np.mean([tg[y, x] == A[y, x] for y, x in zip(yy[s], xx[s])]))
        res["A_agree_family"] = float(np.mean([bool(G.fam_a[tg[y, x] - 1] & G.fam_a[A[y, x] - 1]) for y, x in zip(yy[s], xx[s])]))
    if bd is not None and nd_ids:
        deep = (bd > 0.02).all(0); real = (bd > 0.01).all(0)
        for name, w, ids in (("tetracorder", truth, G.ids_b), ("B", B, G.ids_b), ("A", A, G.ids_a)):
            nd = [i + 1 for i, x in enumerate(ids) if x in nd_ids]
            mm = np.isin(w, nd)
            res[f"nd_{name}"] = dict(pixels=int(mm.sum()), deep_found=int((mm & deep).sum()), deep_total=int(deep.sum()),
                                     frac_real=float(real[mm].mean()) if mm.any() else None)
    return res, truth


# =====================================================================================
# Driver
# =====================================================================================

def run(args) -> dict:
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cube = Cube(Path(args.cube), scale=args.scale, nodata=args.nodata)
    Logger.info(f"cube {cube.lines}x{cube.samples}x{cube.bands} {cube.interleave} {cube.dtype}, "
                f"{cube.wl[0]:.3f}-{cube.wl[-1]:.3f} um, scale {cube.scale:g}")
    valid, vsrc = resolve_valid(cube, args)
    Logger.info(f"usable channels: {valid.sum()}/{cube.bands} (deleted from: {vsrc})")

    expert = Path(args.expert) if args.expert else (
        Path(args.tetracorder_run) / "cmd.lib.setup.t6.00a6" if args.tetracorder_run else
        Path(args.tetracorder_root) / "tetracorder.cmds/tetracorder6.00a.cmds/cmd.lib.setup.t6.00a6")
    dec = TetraDecoder(expert, groups=(), raise_casts=False, variables=args.variables)
    if dec.unresolved:
        raise SystemExit(f"Unresolved expert-system variables {dec.unresolved[:5]}...: pass --variables")
    libs = load_libraries(cube, args, out)
    groups = {g: Group(g, dec, libs, cube.wl, valid, args) for g in args.groups}

    p = dict(fit_min_a=args.fit_min_a, gate_depth=args.gate_depth, mix_dr=args.mix_dr,
             mix_depth=args.mix_depth, mix_twin=args.mix_twin,
             band_depth=ND_BANDS if args.band_depth else None)
    blocks = [(y, min(y + args.block_lines, cube.lines)) for y in range(0, cube.lines, args.block_lines)]
    if args.lines:
        a, b = (int(v) for v in args.lines.split(":"))
        blocks = [(max(y0, a), min(y1, b)) for y0, y1 in blocks if y1 > a and y0 < b]
    ny = (blocks[-1][1] - blocks[0][0]); line0 = blocks[0][0]

    prods = {g: {} for g in groups}; bd_all = None; valid_all = np.zeros((ny, cube.samples), bool)
    state = dict(cube=cube, groups=groups, p=p, valid=valid)

    def collect(r):
        nonlocal bd_all
        s = slice(r["y0"] - line0, r["y1"] - line0)
        valid_all[s] = r["valid"]
        for g in groups:
            for k, v in r[g].items():
                if k not in prods[g]:
                    prods[g][k] = np.zeros(v.shape[:-2] + (ny, cube.samples), v.dtype)
                prods[g][k][..., s, :] = v
        if "band_depth" in r:
            if bd_all is None:
                bd_all = np.zeros((r["band_depth"].shape[0], ny, cube.samples), np.float32)
            bd_all[:, s] = r["band_depth"]

    done = 0
    if args.cores > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(args.cores, initializer=_init_worker, initargs=(state,)) as pool:
            for r in pool.imap_unordered(_process, blocks):
                collect(r); done += 1
                Logger.info(f"block {done}/{len(blocks)}")
    else:
        _init_worker(state)
        for b in blocks:
            collect(_process(b)); done += 1
            Logger.info(f"block {done}/{len(blocks)}")

    # ---- write products
    summary = {"cube": str(cube.path), "shape": [ny, cube.samples, cube.bands], "lines": [line0, line0 + ny],
               "usable_channels": int(valid.sum()), "deleted_from": vsrc, "params": {k: v for k, v in p.items() if k != "band_depth"},
               "groups": {}}
    sub = Cube.__new__(Cube); sub.hdr = cube.hdr if not args.lines else {}
    for g, G in groups.items():
        P = prods[g]; gd = out / f"group{g}"
        tbl = gd / "materials.csv"; gd.mkdir(parents=True, exist_ok=True)
        with open(tbl, "w") as fh:
            fh.write("index,method,id,group,title\n")
            for i, b in enumerate(G.blocks_a, 1): fh.write(f"{i},A,{b['id']},{b['group']},\"{b.get('title','')}\"\n")
            for i, b in enumerate(G.blocks_b, 1): fh.write(f"{i},B,{b['id']},{b['group']},\"{b.get('title','')}\"\n")
        write_envi(gd / "A_rank", P["A_rank"], ["rank1", "rank2", "rank3"], sub, np.uint16, 0, "Method A material index by rank (materials.csv, method A)")
        write_envi(gd / "A_fit", P["A_fit"], ["fit1", "fit2", "fit3"], sub, np.float32, None, "Method A fit F by rank")
        write_envi(gd / "A_abundance", P["A_abund"], ["a1", "a2", "a3"], sub, np.float32, None, "Method A abundance by rank")
        write_envi(gd / "B_rank", P["B_rank"], ["rank1", "rank2", "rank3"], sub, np.uint16, 0, "Method B material index by rank (materials.csv, method B)")
        write_envi(gd / "B_fit", P["B_fit"], ["fit1", "fit2", "fit3"], sub, np.float32, None, "Method B tetracorder-style fit by rank")
        write_envi(gd / "dominant", np.stack([P["A_dom"], P["B_dom"]]), ["A_dominant", "B_dominant"], sub, np.uint16, 0, "dominant material index")
        write_envi(gd / "B_depth", P["B_depth"][None], ["B_dominant_depth"], sub, np.float32, None, "Method B band depth of rank 1")
        write_envi(gd / "secondary", np.concatenate([P["A_sec"], P["B_sec"]]), ["A_sec_rank2", "A_sec_rank3", "B_sec_rank2", "B_sec_rank3"], sub, np.uint16, 0, "accepted secondary index")
        write_envi(gd / "secondary_abundance", np.concatenate([P["A_sec_abund"], P["B_sec_abund"]]), ["A_a2_rank2", "A_a2_rank3", "B_a2_rank2", "B_a2_rank3"], sub, np.float32, None, "secondary abundance a2")

        vm = valid_all
        def cnt(dom, ids):
            vals, c = np.unique(dom[dom > 0], return_counts=True)
            o = np.argsort(-c)[:10]; return {ids[vals[i] - 1]: int(c[i]) for i in o}
        anyA = (P["A_sec"] > 0).any(0); anyB = (P["B_sec"] > 0).any(0)
        gs = dict(
            A_mapped=float((P["A_dom"] > 0)[vm].mean()), B_mapped=float((P["B_dom"] > 0)[vm].mean()),
            A_top=cnt(P["A_dom"], G.ids_a), B_top=cnt(P["B_dom"], G.ids_b),
            A_secondary_frac=float(anyA[vm].mean()), B_secondary_frac=float(anyB[vm].mean()),
            secondary_jaccard=float((anyA & anyB).sum() / max((anyA | anyB).sum(), 1)),
            A_secondary_top=cnt(np.where(P["A_sec"][0] > 0, P["A_sec"][0], P["A_sec"][1]), G.ids_a),
            B_secondary_top=cnt(np.where(P["B_sec"][0] > 0, P["B_sec"][0], P["B_sec"][1]), G.ids_b),
        )
        nd_ids = {b["id"] for b in G.blocks_b if "neodymium" in family(b)}
        if args.tetracorder_run:
            res, _ = compare_with_tetracorder(Path(args.tetracorder_run), g, dec,
                                              P, G, (cube.lines, cube.samples), (line0, line0 + ny),
                                              bd_all if args.band_depth else None, nd_ids)
            gs["vs_tetracorder"] = res
        if args.band_depth and nd_ids:
            deep = (bd_all > 0.02).all(0); real = (bd_all > 0.01).all(0)
            for name, dom, ids in (("A", P["A_dom"], G.ids_a), ("B", P["B_dom"], G.ids_b)):
                nd = [i + 1 for i, x in enumerate(ids) if x in nd_ids]; mm = np.isin(dom, nd)
                gs[f"nd_{name}"] = dict(pixels=int(mm.sum()), deep_found=int((mm & deep).sum()), deep_total=int(deep.sum()),
                                        frac_real=float(real[mm].mean()) if mm.any() else None)
        summary["groups"][g] = gs
    if bd_all is not None:
        write_envi(out / "band_depth_nd", bd_all, list(ND_BANDS), sub, np.float32, None, "continuum band depth, Nd 0.74 / 0.80 um")
        summary["band_depth"] = {"deep_pixels_both_gt_2pct": int((bd_all > 0.02).all(0).sum())}
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    Logger.info(f"done in {summary['seconds']} s -> {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cube", help="ENVI reflectance data file (header must have 'wavelength'; 'fwhm' needed to convolve libraries)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 21])
    ap.add_argument("--tetracorder-root", default=os.environ.get("TETRACORDER_ROOT", str(Path(__file__).resolve().parents[1] / "tetracorder")),
                    help="tetracorder tree (expert system + master libraries); default: repo copy")
    ap.add_argument("--expert", help="expert system file (default: run dir copy, else tetracorder-root)")
    ap.add_argument("--variables", help="cmd.lib.setup.variables file (default: run dir, else VARIABLES/...-default)")
    ap.add_argument("--reflib", help="convolved splib06 on THIS cube's grid (default: convolve now)")
    ap.add_argument("--reslib", help="convolved sprlb06 on THIS cube's grid (default: convolve now)")
    ap.add_argument("--tetracorder-run", help="tetracorder output dir for this cube: reuse its expert system, DELETPTS, and compare")
    ap.add_argument("--deleted-channels", help='1-based tetracorder spec, e.g. "1t4 75t79 218"')
    ap.add_argument("--deleted-um", help='wavelength ranges to delete, e.g. "0.93-0.963 1.327-1.477"')
    ap.add_argument("--scale", type=float, help="multiply raw values by this to get reflectance (default: header / dtype)")
    ap.add_argument("--nodata", type=float)
    ap.add_argument("--temperature-c", type=float, nargs=2, default=DEFAULTS["temperature_c"])
    ap.add_argument("--pressure-bar", type=float, nargs=2, default=DEFAULTS["pressure_bar"])
    ap.add_argument("--fit-min-a", type=float, default=DEFAULTS["fit_min_a"])
    ap.add_argument("--gate-depth", type=float, default=DEFAULTS["gate_depth"])
    ap.add_argument("--mix-dr", type=float, default=DEFAULTS["mix_dr"])
    ap.add_argument("--mix-depth", type=float, default=DEFAULTS["mix_depth"])
    ap.add_argument("--mix-twin", type=float, default=DEFAULTS["mix_twin"])
    ap.add_argument("--no-band-depth", dest="band_depth", action="store_false", help="skip the Nd band-depth check")
    ap.add_argument("--lines", help="process only lines a:b (quick test)")
    ap.add_argument("--block-lines", type=int, default=32)
    ap.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    for n in ("tetrapy.conv", "tetrapy.tetracorder"):
        logging.getLogger(n).setLevel(logging.WARNING if not args.verbose else logging.DEBUG)
    s = run(args)
    print(json.dumps({g: {k: v for k, v in d.items() if not k.endswith("_top")} for g, d in s["groups"].items()}, indent=1))


if __name__ == "__main__":
    main()
