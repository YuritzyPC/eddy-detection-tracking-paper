"""
Eddy detection utilities for the Gulf of California repository.

This module implements a hybrid detector combining:
  1) Normalized Okubo–Weiss + local speed minimum (seed/halo),
  2) Nencioli ring criterion (velocity geometry),
  3) Closed-contour selection with amplitude and geometry filters (CHE11-like).

It is designed to be imported from notebooks, keeping the notebooks clean and
English-only. Functions are typed and documented (NumPy-style docstrings).

Typical usage
-------------
>>> import xarray as xr
>>> from eddydet_detection import Params, detect_day, process_selected_days
>>> ds = xr.open_dataset("anomalias_GC_NeurOST_2010_2024_detrended_allvars.nc")
>>> params = Params()
>>> mask, eddies = detect_day(
...     sla=ds["sla_dtrend2d"].isel(time=0).values,
...     sn=ds["sn_dtrend2d"].isel(time=0).values,
...     ss=ds["ss_dtrend2d"].isel(time=0).values,
...     zeta=ds["zeta_dtrend2d"].isel(time=0).values,
...     ugos=ds["ugosa_dtrend2d"].isel(time=0).values,
...     vgos=ds["vgosa_dtrend2d"].isel(time=0).values,
...     lon=ds["lon"].values,
...     lat=ds["lat"].values,
...     params=params,
... )

The higher-level helper `process_selected_days` builds a tracking-ready
`xarray.Dataset` for a list of time indices, optionally rendering figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Any

import numpy as np
import xarray as xr
from matplotlib.path import Path
from scipy.ndimage import minimum_filter, binary_dilation
from scipy.spatial import ConvexHull
from skimage.measure import label, regionprops
from shapely.geometry import Polygon, Point
from pyproj import Geod

# Optional plotting stack (import lazily inside functions where needed)

# -----------------------------------------------------------------------------
# Parameters and simple containers
# -----------------------------------------------------------------------------

@dataclass
class Params:
    """Algorithm parameters.

    Attributes
    ----------
    Omega : float
        Earth rotation [rad s^-1].
    W_threshold : float
        Lower bound for normalized Okubo–Weiss threshold (more restrictive).
    a_param_km : float
        Nencioli ring radius in km (typical 8–15 km).
    b_param_km : float
        Neighborhood radius for local speed minimum (km).
    eps_speed_p : float
        Fraction of p95(|u|) allowing slack in speed local-min criterion.
    circ_min : float
        Minimum circularity for accepting a closed contour.
    amp_thresh : float
        Minimum amplitude (m) against edge median.
    min_px : int
        Minimum pixels inside contour.
    max_px : int
        Maximum pixels inside contour (for sanity; not always enforced).
    max_diam_km : float
        Maximum allowed diameter (km) based on gulf width.
    dedup_tol_km : float
        Centroid distance tolerance for deduplication.
    """

    Omega: float = 7.2921e-5
    W_threshold: float = -0.02
    a_param_km: float = 10.0
    b_param_km: float = 8.0
    eps_speed_p: float = 0.001
    circ_min: float = 0.6
    amp_thresh: float = 0.025
    min_px: int = 8
    max_px: int = 1000
    max_diam_km: float = 500.0
    dedup_tol_km: float = 10.0


@dataclass
class Eddy:
    """Container for single-eddy properties."""
    time: np.datetime64 | None
    centroid: Tuple[float, float]  # (lat, lon)
    major_axis: float
    minor_axis: float
    diameter: float
    eccentricity: float
    vorticity: float
    ugos: float
    vgos: float
    type: str  # "Cyclonic" or "Anticyclonic"
    contour: np.ndarray  # (N,2) [lon, lat]


# -----------------------------------------------------------------------------
# Geometric helpers
# -----------------------------------------------------------------------------

_geod = Geod(ellps="WGS84")


def grid_spacing_km(lon: np.ndarray, lat: np.ndarray) -> Tuple[float, float]:
    """Return typical grid spacing (dx_km, dy_km)."""
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    lat0 = np.mean(lat)
    dx = np.median(np.abs(dlon)) * 111.0 * np.cos(np.deg2rad(lat0))
    dy = np.median(np.abs(dlat)) * 111.0
    return dx, dy


def circularity(seg: np.ndarray) -> float:
    """Circularity: 4π * area / perimeter^2 computed in km units."""
    lons, lats = seg[:, 0], seg[:, 1]
    lon_km = (lons - lons.mean()) * 111 * np.cos(np.deg2rad(lats.mean()))
    lat_km = (lats - lats.mean()) * 111
    poly = Polygon(np.column_stack((lon_km, lat_km)))
    per = poly.length
    area = poly.area
    return (4 * np.pi * area / per**2) if per > 0 else 0.0


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two (lon,lat) points in kilometers."""
    R = 6371.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def generate_eddy_mask(lat: np.ndarray, lon: np.ndarray, contours: Sequence[np.ndarray]) -> np.ndarray:
    """Return boolean mask True inside any of the provided closed contours."""
    ny, nx = len(lat), len(lon)
    mask = np.zeros((ny, nx), dtype=bool)
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.vstack((LON.ravel(), LAT.ravel())).T
    for seg in contours:
        path = Path(seg)
        mask |= path.contains_points(pts).reshape((ny, nx))
    return mask


def centroid_from_mask(mask: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> Tuple[float, float]:
    """Return centroid (lat, lon) of a boolean mask."""
    ys, xs = np.where(mask)
    return (lat[ys].mean(), lon[xs].mean())


def eddy_stat(mask_bool: np.ndarray, field: np.ndarray) -> float:
    """Masked mean of `field`. Returns NaN if mask is empty."""
    if not mask_bool.any():
        return np.nan
    return np.nanmean(field[mask_bool])


def centroid_core_sla(sla: np.ndarray, lat: np.ndarray, lon: np.ndarray, seg: np.ndarray, *, cyclonic: bool) -> Tuple[float, float]:
    """Core centroid from SLA extreme within polygon `seg` (min for cyclonic, max otherwise)."""
    poly = Polygon(seg)  # seg: (N,2) [lon, lat]
    LON, LAT = np.meshgrid(lon, lat)
    points = np.vstack([LON.ravel(), LAT.ravel()]).T
    mask_int = np.array([poly.contains(Point(p)) for p in points]).reshape(LON.shape)
    vals = sla[mask_int]
    if vals.size == 0:
        c = poly.centroid
        return c.y, c.x
    target = vals.min() if cyclonic else vals.max()
    i0, j0 = np.where(sla == target)
    return lat[i0[0]], lon[j0[0]]


def diameter_from_seg(seg: np.ndarray, lat_c: float, lon_c: float) -> float:
    """Diameter as 2 × median radius from (lat_c, lon_c) to contour vertices (km)."""
    radios = []
    for lon_i, lat_i in seg:
        _, _, d = _geod.inv(lon_c, lat_c, lon_i, lat_i)
        radios.append(d / 1000.0)
    return 2.0 * np.median(radios)


def axes_from_pca_on_seg(seg: np.ndarray) -> Tuple[float, float, float]:
    """Major/minor axes and eccentricity from PCA over the contour (in km)."""
    lons, lats = seg[:, 0], seg[:, 1]
    lon0 = np.mean(lons)
    lat0 = np.mean(lats)
    x = (lons - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
    y = (lats - lat0) * 111.0
    X = np.column_stack([x, y])
    C = np.cov(X, rowvar=False)
    w, _ = np.linalg.eigh(C)
    w = np.sort(w)[::-1]
    major = 2.0 * np.sqrt(max(w[0], 0.0))
    minor = 2.0 * np.sqrt(max(w[1], 0.0))
    ecc = np.sqrt(1 - (minor/major) ** 2) if major > 0 and minor > 0 else 0.0
    return major, minor, ecc


# -----------------------------------------------------------------------------
# Nencioli ring criterion
# -----------------------------------------------------------------------------

def nencioli_ring(mask: np.ndarray, u: np.ndarray, v: np.ndarray, a_pix: int, *, min_ok: int = 6, alpha: float = 0.7) -> np.ndarray:
    """Apply Nencioli ring criterion on boolean `mask` using velocity fields.

    A pixel is kept if, at the ring of radius `a_pix` sampled at 8 directions:
      - Tangential component `vt` is stronger than center and consistent in sign
        (>= `min_ok` of 8 directions), and
      - Radial component `vr` is bounded by |vr| < alpha * |vt|.
    """
    ny, nx = mask.shape
    refined = np.zeros_like(mask, dtype=bool)
    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for i in range(a_pix, ny - a_pix):
        for j in range(a_pix, nx - a_pix):
            if not mask[i, j]:
                continue
            vt_signs: List[int] = []
            ok = 0
            for dy, dx in dirs:
                ii = i + dy * a_pix
                jj = j + dx * a_pix
                rr = np.array([dx, dy], dtype=float)
                rr /= np.hypot(rr[0], rr[1])
                t_hat = np.array([-rr[1], rr[0]])
                vt = u[ii, jj] * t_hat[0] + v[ii, jj] * t_hat[1]
                vr = u[ii, jj] * rr[0] + v[ii, jj] * rr[1]
                vt_c = u[i, j] * t_hat[0] + v[i, j] * t_hat[1]
                if (abs(vt) > abs(vt_c)) and (abs(vr) < alpha * abs(vt)):
                    vt_signs.append(int(np.sign(vt)) if vt != 0 else 0)
                    ok += 1
            if ok >= min_ok and vt_signs:
                if abs(np.sum(vt_signs)) >= (min_ok - 1):  # allow at most one mismatch
                    refined[i, j] = True
    return refined


# -----------------------------------------------------------------------------
# Main detection for a single day
# -----------------------------------------------------------------------------

def _sla_contours(sla: np.ndarray, lon: np.ndarray, lat: np.ndarray, *, nlev: int = 50) -> Tuple[Sequence[float], Sequence[Sequence[np.ndarray]]]:
    """Return contour levels and segments using matplotlib (off-screen)."""
    import matplotlib
    matplotlib.use("Agg")  # off-screen backend
    import matplotlib.pyplot as plt

    vmax = float(np.nanmax(np.abs(sla)))
    levels = np.linspace(-vmax, vmax, nlev)
    fig, ax = plt.subplots()
    cs = ax.contour(lon, lat, sla, levels=levels)
    levels_cs, allsegs = cs.levels, cs.allsegs
    plt.close(fig)
    return levels_cs, allsegs


def detect_day(
    *,
    sla: np.ndarray,
    sn: np.ndarray,
    ss: np.ndarray,
    zeta: np.ndarray,
    ugos: np.ndarray,
    vgos: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    params: Params,
    time_value: np.datetime64 | None = None,
) -> Tuple[np.ndarray, List[Eddy]]:
    """Run hybrid detection for a single day.

    Returns
    -------
    mask_day : (ny, nx) bool
        Eddy mask (True inside any accepted contour)
    eddies : list[Eddy]
        Extracted eddies with properties and original contour.
    """
    # --- Stage 1: normalized Okubo–Weiss + local speed minimum
    W = sn**2 + ss**2 - zeta**2
    f = 2 * params.Omega * np.sin(np.deg2rad(lat))
    Wn = W / (f[:, None] ** 2)

    thr_dyn = np.nanpercentile(Wn, 10)
    thr = min(params.W_threshold, thr_dyn)
    candidate = Wn < thr

    speed = np.sqrt(ugos**2 + vgos**2)
    dx_km, dy_km = grid_spacing_km(lon, lat)
    pix_km = float(np.mean([dx_km, dy_km]))
    b_pix = max(1, int(round(params.b_param_km / pix_km)))
    a_pix = max(1, int(round(params.a_param_km / pix_km)))

    eps = params.eps_speed_p * np.nanpercentile(speed, 95)
    local_min = speed <= (minimum_filter(speed, size=b_pix) + eps)
    halo_mask = binary_dilation(
        candidate & local_min, structure=np.ones((2 * b_pix + 1, 2 * b_pix + 1))
    )

    # --- Stage 2: Nencioli ring
    refined_mask = nencioli_ring(halo_mask, ugos, vgos, a_pix=a_pix, min_ok=6, alpha=0.7)

    # --- Stage 3: closed contours (CHE11-like) and filters
    levels_cs, allsegs = _sla_contours(sla, lon, lat)

    conts: List[Tuple[float, np.ndarray]] = []
    for lvl, segs in zip(levels_cs, allsegs):
        for seg in segs:
            if seg.shape[0] < 3 or not np.allclose(seg[0], seg[-1], atol=1e-3):
                continue
            if circularity(seg) >= params.circ_min:
                conts.append((lvl, seg))

    # remove "deformed parents"
    circs = [circularity(s) for _, s in conts]
    kept: List[Tuple[float, np.ndarray]] = []
    for i, (lvl_i, seg_i) in enumerate(conts):
        pi = Path(seg_i)
        drop = False
        for j, (_, seg_j) in enumerate(conts):
            if i == j:
                continue
            if pi.contains_path(Path(seg_j)) and circs[j] > circs[i] * 1.2:
                drop = True
                break
        if not drop:
            kept.append((lvl_i, seg_i))
    conts = kept

    # group nested families
    groups: List[List[Tuple[float, np.ndarray]]] = []
    used = set()
    for i, (lvl_i, seg_i) in enumerate(conts):
        if i in used:
            continue
        Pi = Path(seg_i)
        fam = [(lvl_i, seg_i)]
        used.add(i)
        for j, (lvl_j, seg_j) in enumerate(conts):
            if j in used:
                continue
            if Pi.contains_path(Path(seg_j)):
                fam.append((lvl_j, seg_j))
                used.add(j)
        groups.append(fam)

    # select best segment per family by amplitude
    LON2D, LAT2D = np.meshgrid(lon, lat)
    pts_flat = np.column_stack((LON2D.ravel(), LAT2D.ravel()))

    selected: List[np.ndarray] = []
    for fam in groups:
        best_A = -np.inf
        best_seg = None
        for lvl, seg in fam:
            path = Path(seg)
            mask_int = path.contains_points(pts_flat).reshape(sla.shape)
            if not mask_int.any():
                continue
            idx_i = [np.argmin(np.abs(lat - lat_i)) for _, lat_i in seg]
            idx_j = [np.argmin(np.abs(lon - lon_i)) for lon_i, _ in seg]
            h0 = np.nanmean(sla[idx_i, idx_j])
            A = max(np.nanmax(sla[mask_int]) - h0, h0 - np.nanmin(sla[mask_int]))
            if A > best_A:
                best_A, best_seg = A, seg
        if best_seg is not None and best_A >= params.amp_thresh:
            selected.append(best_seg)

    # final filters
    final_segs: List[np.ndarray] = []
    mask_contours = np.zeros_like(sla, dtype=bool)
    for seg in selected:
        path = Path(seg)
        mask_int = path.contains_points(pts_flat).reshape(sla.shape)
        if mask_int.sum() < params.min_px:
            continue
        idx_i = [np.argmin(np.abs(lat - lat_i)) for _, lat_i in seg]
        idx_j = [np.argmin(np.abs(lon - lon_i)) for lon_i, _ in seg]
        h0 = np.nanmean(sla[idx_i, idx_j])
        A = max(np.nanmax(sla[mask_int]) - h0, h0 - np.nanmin(sla[mask_int]))
        if A < params.amp_thresh:
            continue
        hull_pts = seg[ConvexHull(seg).vertices]
        dmax = max(
            haversine(p1[0], p1[1], p2[0], p2[1]) for p1 in hull_pts for p2 in hull_pts
        )
        if dmax > params.max_diam_km:
            continue
        final_segs.append(seg)
        mask_contours |= mask_int

    # leftover coherent blobs from refined_mask outside contours
    leftover = refined_mask & (~mask_contours)
    for prop in regionprops(label(leftover)):
        if prop.area < 10:
            continue
        ys, xs = prop.coords[:, 0], prop.coords[:, 1]
        pts2 = np.column_stack((lon[xs], lat[ys]))
        try:
            seg2 = pts2[ConvexHull(pts2).vertices]
            final_segs.append(seg2)
        except Exception:
            pass

    # remove redundant nested contours
    filtered_segs: List[np.ndarray] = [
        seg
        for seg in final_segs
        if not any(Path(o).contains_path(Path(seg)) and not np.array_equal(o, seg) for o in final_segs)
    ]

    # deduplicate by centroid distance
    final_contours: List[np.ndarray] = []
    seen_centroids: List[Tuple[float, float]] = []  # (lon, lat)
    for seg in filtered_segs:
        mask_e = Path(seg).contains_points(pts_flat).reshape(sla.shape)
        lat_c, lon_c = centroid_from_mask(mask_e, lon=lon, lat=lat)
        if all(haversine(lon_c, lat_c, lon0, lat0) > params.dedup_tol_km for lon0, lat0 in seen_centroids):
            final_contours.append(seg)
            seen_centroids.append((lon_c, lat_c))

    # build eddy mask and extract properties
    mask_day = generate_eddy_mask(lat, lon, final_contours)

    eddies: List[Eddy] = []
    for seg in final_contours:
        mask_e = Path(seg).contains_points(pts_flat).reshape(sla.shape)
        vort = eddy_stat(mask_e, zeta)
        ug = eddy_stat(mask_e, ugos)
        vg = eddy_stat(mask_e, vgos)
        eddy_type = "Cyclonic" if (vort > 0) else "Anticyclonic"
        lat_c, lon_c = centroid_core_sla(sla, lat, lon, seg, cyclonic=(eddy_type == "Cyclonic"))
        diameter_km = diameter_from_seg(seg, lat_c, lon_c)
        major, minor, ecc = axes_from_pca_on_seg(seg)
        eddies.append(
            Eddy(
                time=time_value,
                centroid=(lat_c, lon_c),
                major_axis=major,
                minor_axis=minor,
                diameter=diameter_km,
                eccentricity=ecc,
                vorticity=vort,
                ugos=ug,
                vgos=vg,
                type=eddy_type,
                contour=seg,
            )
        )

    return mask_day, eddies


# -----------------------------------------------------------------------------
# Tracking dataset builder and optional plotting
# -----------------------------------------------------------------------------

def process_selected_days(
    ds: xr.Dataset,
    time_indices: Sequence[int],
    params: Params,
    *,
    show_figs: bool = False,
) -> xr.Dataset:
    """Process selected `time_indices` from `ds` and return tracking-ready Dataset.

    Notes
    -----
    Expected variables in `ds` (names follow the notebooks):
      - `sla_dtrend2d`, `sn_dtrend2d`, `ss_dtrend2d`, `zeta_dtrend2d`,
        `ugosa_dtrend2d`, `vgosa_dtrend2d`, and coordinates `lat`, `lon`, `time`.
    If `show_figs` is True, draws simple cartopy maps for each processed day.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    lat = ds["lat"].values
    lon = ds["lon"].values
    n_time = ds.time.size

    # containers
    mask_3d = np.zeros((n_time, len(lat), len(lon)), dtype=bool)
    eddies_by_day: Dict[Any, List[Eddy]] = {ts.item(): [] for ts in ds.time.values}

    for t_idx in time_indices:
        ts = ds.time.values[t_idx]
        date = np.datetime64(ts)
        mask_day, eddies = detect_day(
            sla=ds["sla_dtrend2d"].isel(time=t_idx).values,
            sn=ds["sn_dtrend2d"].isel(time=t_idx).values,
            ss=ds["ss_dtrend2d"].isel(time=t_idx).values,
            zeta=ds["zeta_dtrend2d"].isel(time=t_idx).values,
            ugos=ds["ugosa_dtrend2d"].isel(time=t_idx).values,
            vgos=ds["vgosa_dtrend2d"].isel(time=t_idx).values,
            lon=lon,
            lat=lat,
            params=params,
            time_value=date,
        )
        mask_3d[t_idx] = mask_day
        eddies_by_day[ts.item()] = eddies

        if show_figs:
            try:
                _plot_day_quick(lon, lat, ds["sla_dtrend2d"].isel(time=t_idx).values, eddies, date)
            except Exception:
                pass

    # build tracking dataset (pad to max number of eddies per day)
    max_eddies = max(len(v) for v in eddies_by_day.values()) or 1
    eddy_idx = np.arange(max_eddies)

    cent_lat = np.full((n_time, max_eddies), np.nan)
    cent_lon = np.full((n_time, max_eddies), np.nan)
    diam_km = np.full((n_time, max_eddies), np.nan)
    major_km = np.full((n_time, max_eddies), np.nan)
    minor_km = np.full((n_time, max_eddies), np.nan)
    ecc_arr = np.full((n_time, max_eddies), np.nan)
    vort_arr = np.full((n_time, max_eddies), np.nan)
    ug_arr = np.full((n_time, max_eddies), np.nan)
    vg_arr = np.full((n_time, max_eddies), np.nan)
    type_arr = np.full((n_time, max_eddies), "", dtype=object)

    times = ds.time.values
    for i, ts in enumerate(times):
        for j, eddy in enumerate(eddies_by_day[ts.item()]):
            cent_lat[i, j] = eddy.centroid[0]
            cent_lon[i, j] = eddy.centroid[1]
            major_km[i, j] = eddy.major_axis
            minor_km[i, j] = eddy.minor_axis
            diam_km[i, j] = eddy.diameter
            ecc_arr[i, j] = eddy.eccentricity
            vort_arr[i, j] = eddy.vorticity
            ug_arr[i, j] = eddy.ugos
            vg_arr[i, j] = eddy.vgos
            type_arr[i, j] = eddy.type

    ds_out = xr.Dataset(
        {
            "centroid_lat": ("time", "eddy"),
            "centroid_lon": ("time", "eddy"),
        }
    )
    # Build variables explicitly to avoid tuple confusion
    ds_out["centroid_lat"] = ("time", "eddy"), cent_lat
    ds_out["centroid_lon"] = ("time", "eddy"), cent_lon
    ds_out["diameter_km"] = ("time", "eddy"), diam_km
    ds_out["major_axis_km"] = ("time", "eddy"), major_km
    ds_out["minor_axis_km"] = ("time", "eddy"), minor_km
    ds_out["eccentricity"] = ("time", "eddy"), ecc_arr
    ds_out["vorticity"] = ("time", "eddy"), vort_arr
    ds_out["ugos"] = ("time", "eddy"), ug_arr
    ds_out["vgos"] = ("time", "eddy"), vg_arr
    ds_out["type"] = ("time", "eddy"), type_arr

    ds_out["mask_eddies"] = ("time", "lat", "lon"), mask_3d

    # optional reference fields (non-detrended anomalies if present)
    if {"sla_anomaly", "sn_anomaly", "ss_anomaly", "zeta_anomaly"}.issubset(ds.variables):
        ds_out["ssha"] = ("time", "lat", "lon"), ds.sla_anomaly.values
        ds_out["okubo_weiss"] = (
            ("time", "lat", "lon"),
            ds.sn_anomaly.values**2 + ds.ss_anomaly.values**2 - ds.zeta_anomaly.values**2,
        )

    ds_out = ds_out.assign_coords({"time": ds.time.values, "lat": lat, "lon": lon, "eddy": eddy_idx})
    return ds_out


# -----------------------------------------------------------------------------
# Quick plotting (optional gallery figures)
# -----------------------------------------------------------------------------

def _plot_day_quick(lon: np.ndarray, lat: np.ndarray, sla: np.ndarray, eddies: Sequence[Eddy], date: np.datetime64) -> None:
    """Simple cartopy plot useful for the repo gallery (not for the paper)."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    cf = ax.contourf(lon, lat, sla, levels=20, cmap="RdBu_r", transform=ccrs.PlateCarree(), zorder=1)
    ax.coastlines(resolution="10m", zorder=2)
    ax.add_feature(cfeature.LAND, color="black", zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    for eddy in eddies:
        seg = eddy.contour
        lat_c, lon_c = eddy.centroid
        radius_km = eddy.diameter / 2
        ax.plot(seg[:, 0], seg[:, 1], "--", linewidth=2, transform=ccrs.PlateCarree(), zorder=3)
        # reference circle with same diameter
        theta = np.linspace(0, 2 * np.pi, 200)
        dx = radius_km * np.cos(theta)
        dy = radius_km * np.sin(theta)
        lon_circ = lon_c + dx / (111.0 * np.cos(np.deg2rad(lat_c)))
        lat_circ = lat_c + dy / 111.0
        ax.plot(lon_circ, lat_circ, ":", lw=1, transform=ccrs.PlateCarree(), zorder=3)
        ax.plot(lon_c, lat_c, "x", transform=ccrs.PlateCarree(), zorder=4)

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("SSHA (m)")
    ax.set_title(f"Detected Eddies — {np.datetime_as_string(date, unit='D')}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
