"""
Whole-shape library fit for narrow-band absorbers (default: Nd oxide, USGS GDS34).

Unlike the two-window band depth in ``mixmap`` (one 0.74 and one 0.80 um depth, each from its own
local continuum), this fits the *entire* continuum-removed library band pattern over 0.50-0.95 um
at once:

1. Continuum: upper convex hull of the pixel spectrum over [lo, hi], computed per pixel on the
   usable (non-deleted) channels. Optional channel masks (O2-A by default) are dropped from the
   fit but still used for the hull.
2. Absorption vectors: ``y = 1 - CR(pixel)``, ``t = 1 - CR(library)``, the library convolved to the
   cube's wavelengths/FWHM.
3. Fit ``y = a t + P(lambda)`` by least squares, where ``P`` is a low-order polynomial baseline
   (default quadratic, ``--baseline-deg``). The baseline absorbs broad Fe/slope residue the hull
   leaves behind; without it (``--baseline-deg -1``) broad features project onto the library's
   broad CR shape and ~30% of a desert scene exceeds 3.5%.
   **depth = a * max(t)**, i.e. the library's deepest band scaled to the pixel (fraction, 0.05 = 5%).
4. Fit quality: partial R^2 of the library term,
   ``r2 = 1 - |y - a t - P|^2 / |y - P_only|^2`` (how much of what the baseline cannot explain the
   Nd shape does explain).

Detections must pass all three thresholds:

- ``depth >= --min-depth`` (default 0.035),
- ``r2 >= --min-r2`` (default 0.5): the Nd shape has to explain the absorption, not just overlap it,
- ``rfl_max >= --min-rfl`` (default 0.05, brightest fit channel): depth is relative to the
  continuum, so very dark pixels (shadow, water) reach large "depths" from noise alone.

Outputs (in --out): ``<name>_depth`` and ``<name>_r2`` ENVI float32 rasters (all pixels, unthresholded),
``<name>_hits.csv`` (detections only: line, sample, depth, r2, rfl_max, easting, northing) and
``<name>.json`` (counts after each threshold).

Inputs: an ENVI cube (wavelength + fwhm in header) or a netCDF4 reflectance file with
``reflectance (bands, y, x)``, ``wavelength`` and ``fwhm`` variables (EMIT/AVIRIS-5 L2A ORT layout).
netCDF4 is used if installed, else the bundled pure-python reader (tetrapy/h5lite.py).

Example
-------
::

    python tetrapy/ndfit.py AV520261106t201609_010_L2A_OE_036fa792_RFL_ORT.nc --out ndfit/ \\
        --deleted-um "0.38-0.405 0.93-0.963 1.11-1.164 1.327-1.477 1.770-1.968 2.455-2.50"
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tetrapy.conv.specpr import SpecprFile
except ImportError:          # standalone use without the full tetrapy install (see mixmap.py)
    import types as _types
    _pkg = _types.ModuleType("tetrapy"); _pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules.setdefault("tetrapy", _pkg)
    from tetrapy.conv.specpr import SpecprFile
from tetrapy.conv.convolve import Convolver

Logger = logging.getLogger("tetrapy.ndfit")
ROOT = Path(__file__).resolve().parents[1]

# name: (master library, spectrum record, wavelength record, fwhm record) in tetracorder/sl1/usgs/library06.conv
TARGETS = {
    "nd_oxide": ("splib06b", 6623, 6, 17),         # Neodymium_Oxide GDS34 W1R1Ba AREF
    "monazite957": ("sprlb06b", 2044, 28, 34),     # Monazite_REE GDS957 New Mex
    "monazite947": ("sprlb06b", 2038, 28, 34),     # Monazite_REE GDS947 Calif
}
DEFAULT_MASK = "0.757-0.771"                       # O2-A


def parse_ranges(s):
    out = []
    for tok in (s or "").replace(",", " ").split():
        a, b = tok.split("-"); out.append((float(a), float(b)))
    return out


# ------------------------------------------------------------------------------------- I/O
class NCCube:
    """netCDF4 L2A reflectance: reflectance(bands, y, x), wavelength, fwhm (nm)."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            import netCDF4
            ds = netCDF4.Dataset(path); ds.set_auto_mask(False)
            self._r = ds["reflectance"]; get = lambda k: np.asarray(ds[k][:])
            self.easting = get("easting") if "easting" in ds.variables else None
            self.northing = get("northing") if "northing" in ds.variables else None
        except ImportError:
            from tetrapy.h5lite import H5
            h = H5(str(path)); D = {h.names.get(a): h.reader(a) for a in h.datasets()}
            self._r = D["reflectance"]; get = lambda k: D[k].read_all()
            self.easting = get("easting") if "easting" in D else None
            self.northing = get("northing") if "northing" in D else None
        wl, fw = get("wavelength").astype(float), get("fwhm").astype(float)
        if wl.max() > 100: wl, fw = wl / 1000, fw / 1000
        self.wl, self.fwhm = wl, fw
        self.bands, self.lines, self.samples = self._r.shape
        self.map_info = None

    def block(self, y0, y1, chans):
        """(lines, samples, len(chans)) float32 for a contiguous channel range covering chans."""
        c0, c1 = int(chans.min()), int(chans.max()) + 1
        x = np.asarray(self._r[c0:c1, y0:y1, :], dtype=np.float32)
        return x[chans - c0].transpose(1, 2, 0)


class ENVICube:
    def __init__(self, path):
        from tetrapy.mixmap import Cube
        self.c = Cube(Path(path)); self.path = Path(path)
        self.wl, self.fwhm = self.c.wl, self.c.fwhm
        self.bands, self.lines, self.samples = self.c.bands, self.c.lines, self.c.samples
        self.easting = self.northing = None
        self.map_info = self.c.hdr.get("map info")

    def block(self, y0, y1, chans):
        refl, bad = self.c.block(y0, y1)
        x = refl[..., chans]; x[bad] = np.nan
        return x


def open_cube(path):
    p = Path(path)
    return NCCube(p) if p.suffix.lower() in (".nc", ".nc4", ".h5") else ENVICube(p)


def write_envi(path, arr, name, cube):
    arr = np.ascontiguousarray(arr[None].astype(np.float32)); arr.tofile(path)
    h = ["ENVI", f"samples = {arr.shape[2]}", f"lines = {arr.shape[1]}", "bands = 1", "header offset = 0",
         "file type = ENVI Standard", "data type = 4", "interleave = bsq", "byte order = 0",
         f"band names = {{{name}}}", "data ignore value = nan"]
    if cube.map_info:
        mi = cube.map_info; h.append(f"map info = {{{', '.join(mi) if isinstance(mi, list) else mi}}}")
    elif cube.easting is not None and cube.northing is not None:
        de = cube.easting[1] - cube.easting[0]; dn = cube.northing[0] - cube.northing[1]
        h.append(f"map info = {{UTM, 1, 1, {cube.easting[0] - de / 2}, {cube.northing[0] + dn / 2}, {de}, {dn}, 11, North, WGS-84}}"
                 "  ; zone assumed 11N, check")
    Path(str(path) + ".hdr").write_text("\n".join(h) + "\n")


# ------------------------------------------------------------------------------------- core
def library_spectrum(target, wl, fwhm, root=ROOT / "tetracorder/sl1/usgs/library06.conv"):
    lib, rec, wrec, frec = TARGETS[target]
    L = SpecprFile.open(Path(root) / lib)
    s, w, f = (L.read_spectrum(r).astype(float) for r in (rec, wrec, frec))
    return Convolver(wl, fwhm).convolve(w, f, s).astype(float)


def hull_cr(w, Y):
    """Continuum removal by upper convex hull, vectorised over leading dims of Y (..., n).
    NaNs are not allowed (caller masks bad pixels)."""
    shp = Y.shape; Y2 = Y.reshape(-1, shp[-1]); out = np.empty_like(Y2)
    n = len(w)
    for i, y in enumerate(Y2):
        hull = [0]
        for k in range(1, n):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (y[b] - y[a]) * (w[k] - w[a]) <= (y[k] - y[a]) * (w[b] - w[a]): hull.pop()
                else: break
            hull.append(k)
        out[i] = y / np.interp(w, w[hull], y[hull])
    return out.reshape(shp)


def hull_cr_fast(w, Y):
    """Vectorised upper convex hull via Andrew's monotone chain, run on all pixels at once.

    Points are visited left to right; each pixel keeps its own stack (``stk``, depth ``top``).
    Before pushing point k, pop while the last two stacked points and k make a non-right turn.
    The pops for a given k are bounded by the stack depth, so the inner loop runs only while some
    pixel still needs a pop. Exact (identical to :func:`hull_cr`)."""
    Y = np.asarray(Y, np.float64); P, n = Y.reshape(-1, Y.shape[-1]).shape
    Y2 = Y.reshape(P, n); r = np.arange(P)
    stk = np.zeros((P, n), np.int32); top = np.ones(P, np.int32)       # stk[:, :top] = hull so far
    for k in range(1, n):
        while True:
            can = top >= 2
            if not can.any(): break
            a = stk[r, np.maximum(top - 2, 0)]; b = stk[r, top - 1]
            cross = (Y2[r, b] - Y2[r, a]) * (w[k] - w[a]) - (Y2[:, k] - Y2[r, a]) * (w[b] - w[a])
            pop = can & (cross <= 0)
            if not pop.any(): break
            top -= pop
        stk[r, top] = k; top += 1
    on = np.zeros((P, n), bool)
    for j in range(n):
        m = top > j
        if not m.any(): break
        on[r[m], stk[m, j]] = True
    return (Y2 / _interp_rows(w, Y2, on)).reshape(Y.shape)


def _interp_rows(w, Y, on):
    """Row-wise linear interpolation of Y through the columns flagged in on."""
    P, n = Y.shape
    idx = np.where(on, np.arange(n)[None], -1)
    left = np.maximum.accumulate(idx, axis=1)                      # last vertex at or before j
    idx_r = np.where(on, np.arange(n)[None], n)
    right = np.minimum.accumulate(idx_r[:, ::-1], axis=1)[:, ::-1]  # next vertex at or after j
    r = np.arange(P)[:, None]
    yl, yr = Y[r, left], Y[r, right]; wl_, wr_ = w[left], w[right]
    t = np.where(right > left, (w[None] - wl_) / np.where(right > left, wr_ - wl_, 1), 0.0)
    return yl + (yr - yl) * t


def baseline(w, deg):
    if deg < 0: return np.zeros((w.size, 0))
    x = (w - w.mean()) / (np.ptp(w) / 2)
    return np.stack([x ** k for k in range(deg + 1)], 1)


def fit(CR, tvec, w, keep, deg=2):
    """CR: (P, n) pixel continuum-removed; tvec: (n,) library absorption; keep: (n,) fit channels.
    Returns depth (= a * max t) and the partial R^2 of the library term."""
    y = (1 - CR[:, keep]).T                                   # (k, P)
    B = baseline(w, deg)[keep]; M = np.c_[tvec[keep], B]
    coef = np.linalg.lstsq(M, y, rcond=None)[0]
    res = y - M @ coef
    rb = y - B @ np.linalg.lstsq(B, y, rcond=None)[0] if B.shape[1] else y
    r2 = 1 - (res ** 2).sum(0) / np.maximum((rb ** 2).sum(0), 1e-12)
    return coef[0] * tvec.max(), r2


def run(args):
    t0 = time.time()
    cube = open_cube(args.cube)
    wl = cube.wl
    usable = np.ones(wl.size, bool)
    for a, b in parse_ranges(args.deleted_um): usable &= ~((wl >= a) & (wl <= b))
    chans = np.flatnonzero(usable & (wl >= args.lo) & (wl <= args.hi))
    w = wl[chans]
    keep = np.ones(w.size, bool)
    for a, b in parse_ranges(args.mask_um): keep &= ~((w >= a) & (w <= b))
    lib = library_spectrum(args.target, wl, cube.fwhm)[chans]
    tvec = 1 - hull_cr(w, lib[None])[0]
    Logger.info(f"{cube.path.name}: {cube.lines}x{cube.samples}, {chans.size} fit channels "
                f"{w[0]:.3f}-{w[-1]:.3f} um, target {args.target} (max CR depth {tvec.max():.3f})")
    y0s = range(0, cube.lines, args.block_lines)
    D = np.full((cube.lines, cube.samples), np.nan, np.float32); R2 = np.full_like(D, np.nan); BR = np.full_like(D, np.nan)
    for y0 in y0s:
        y1 = min(y0 + args.block_lines, cube.lines)
        X = cube.block(y0, y1, chans)
        good = np.isfinite(X).all(-1) & (X > 0).all(-1)
        if good.any():
            Xg = X[good]
            CR = np.concatenate([hull_cr_fast(w, Xg[i:i + 8192]) for i in range(0, len(Xg), 8192)])
            d, r2 = fit(CR, tvec, w, keep, args.baseline_deg)
            D[y0:y1][good] = d; R2[y0:y1][good] = r2; BR[y0:y1][good] = Xg.max(-1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.cube).name.split("_L2A")[0] if "_L2A" in Path(args.cube).name else Path(args.cube).stem
    name = f"{stem}_{args.target}"
    write_envi(out / f"{name}_depth", D, f"{args.target} fit depth", cube)
    write_envi(out / f"{name}_r2", R2, f"{args.target} fit r2", cube)
    pass_d = D >= args.min_depth; pass_r = R2 >= args.min_r2; pass_b = BR >= args.min_rfl
    det = pass_d & pass_r & pass_b
    ys, xs = np.nonzero(det)
    o = np.argsort(-D[ys, xs]); ys, xs = ys[o], xs[o]
    with open(out / f"{name}_hits.csv", "w") as f:
        f.write("line,sample,depth,r2,rfl_max,easting,northing\n")
        for y, x in zip(ys, xs):
            e = cube.easting[x] if cube.easting is not None else np.nan
            n = cube.northing[y] if cube.northing is not None else np.nan
            f.write(f"{y},{x},{D[y, x]:.5f},{R2[y, x]:.4f},{BR[y, x]:.4f},{e:.2f},{n:.2f}\n")
    v = np.isfinite(D)
    summ = dict(cube=str(args.cube), target=args.target, baseline_deg=args.baseline_deg, window_um=[args.lo, args.hi], mask_um=args.mask_um,
                fit_channels=int(chans.size), valid_pixels=int(v.sum()),
                depth_pct=dict(zip(["p50", "p99", "p99.9", "p99.99", "max"],
                                   [float(x) for x in np.r_[np.nanpercentile(D, [50, 99, 99.9, 99.99]), np.nanmax(D)]])),
                thresholds=dict(min_depth=args.min_depth, min_r2=args.min_r2, min_rfl=args.min_rfl),
                n_pass_depth=int(pass_d.sum()), n_pass_depth_rfl=int((pass_d & pass_b).sum()),
                n_pass_depth_r2=int((pass_d & pass_r).sum()), n_detections=int(len(ys)),
                seconds=round(time.time() - t0, 1))
    (out / f"{name}.json").write_text(json.dumps(summ, indent=1))
    Logger.info(f"done in {summ['seconds']} s: {summ['n_pass_depth']} px depth >= {args.min_depth}, "
                f"{summ['n_detections']} also pass r2 >= {args.min_r2} and rfl >= {args.min_rfl}")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cube", help="netCDF4 L2A reflectance (.nc) or ENVI cube")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="nd_oxide", choices=sorted(TARGETS))
    ap.add_argument("--lo", type=float, default=0.50)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--deleted-um", default="0.38-0.405 0.93-0.963 1.11-1.164 1.327-1.477 1.770-1.968 2.455-2.50")
    ap.add_argument("--mask-um", default=DEFAULT_MASK, help="ranges kept for the hull but dropped from the fit")
    ap.add_argument("--baseline-deg", type=int, default=2, help="polynomial baseline fitted with the library (-1 = none)")
    ap.add_argument("--min-depth", type=float, default=0.035, help="detection: fit depth >= this (fraction)")
    ap.add_argument("--min-r2", type=float, default=0.5, help="detection: partial R^2 of the library term >= this")
    ap.add_argument("--min-rfl", type=float, default=0.05, help="detection: brightest fit-channel reflectance >= this")
    ap.add_argument("--block-lines", type=int, default=64)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    print(json.dumps(run(args), indent=1))


if __name__ == "__main__":
    main()
