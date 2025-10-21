"""
Tracking utilities for eddy detections in the Gulf of California project.

This module links daily eddy detections into temporally coherent tracks using
an inertial cost (predicted position), diameter change, vorticity consistency,
and type consistency, while tolerating gaps up to a configurable number of days.

Typical usage
-------------
>>> import xarray as xr
>>> from eddydet.tracking import TrackParams, build_detections_dict, convert_to_states, track_eddies, tracks_to_dataframe
>>> ds = xr.open_dataset("remolinos_completo_refinado2_2cm.nc")
>>> dets = build_detections_dict(ds)
>>> states = convert_to_states(dets)
>>> params = TrackParams()
>>> tracks = track_eddies(states, dist_max=params.dist_max_km, cost_threshold=params.cost_threshold, allowed_gap=params.allowed_gap_days)
>>> df = tracks_to_dataframe(tracks)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class TrackParams:
    """Tuning parameters for the tracking algorithm.

    Attributes
    ----------
    dist_max_km : float
        Maximum distance scale (km) used for distance cost normalization.
    cost_threshold : float
        Maximum total cost to accept a match; otherwise a new track is started.
    allowed_gap_days : int
        Maximum allowed gap (days) between detections for a track continuation.
    min_track_length : int
        Minimum number of detections to keep a track in post-filtering.
    alpha : float
        Weight for distance term in the matching cost.
    beta : float
        Weight for diameter-change term in the matching cost.
    gamma : float
        Weight for vorticity-change term in the matching cost.
    delta : float
        Penalty for type mismatch (Cyclonic vs Anticyclonic). Large value to discourage.
    gap_penalty_per_day : float
        Multiplicative penalty applied when bridging gaps greater than 1 day.
    """

    dist_max_km: float = 50.0
    cost_threshold: float = 1.2
    allowed_gap_days: int = 10
    min_track_length: int = 14
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 1.0
    delta: float = 100.0
    gap_penalty_per_day: float = 0.1


# -----------------------------------------------------------------------------
# Geometry & helpers
# -----------------------------------------------------------------------------

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two geographic points.

    Returns NaN if any input is NaN; clamps numerical roundoff.
    """
    if any(np.isnan([lon1, lat1, lon2, lat2])):
        return np.nan
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class EddyState:
    """Single-day eddy state used by the tracker.

    day : numpy.datetime64
    centroid : tuple[float, float]  # (lat, lon)
    diameter : float                # km
    major_axis : float              # km
    minor_axis : float              # km
    eccentricity : float
    vorticity : float
    eddy_type : str                 # "Cyclonic" or "Anticyclonic"
    velocity : tuple[float, float] = (0.0, 0.0)  # (vx, vy) in km/day
    track_id : Optional[int] = None
    """

    day: np.datetime64
    centroid: tuple[float, float]
    diameter: float
    major_axis: float
    minor_axis: float
    eccentricity: float
    vorticity: float
    eddy_type: str
    velocity: tuple[float, float] = (0.0, 0.0)
    track_id: Optional[int] = None

    def __repr__(self) -> str:  # pretty, compact
        lat, lon = self.centroid
        return (
            f"EddyState(day={self.day.astype('datetime64[D]')}, centroid=({lat:.2f},{lon:.2f}), "
            f"diam={self.diameter:.1f}km, major={self.major_axis:.1f}km, minor={self.minor_axis:.1f}km, "
            f"ecc={self.eccentricity:.2f}, vort={self.vorticity:.2e}, type={self.eddy_type}, "
            f"vel=({self.velocity[0]:.1f},{self.velocity[1]:.1f}), track={self.track_id})"
        )


# -----------------------------------------------------------------------------
# Core tracking math
# -----------------------------------------------------------------------------

def update_velocity(eddy_prev: EddyState, eddy_current: EddyState, gap: Optional[int] = None) -> None:
    """Estimate (vx, vy) in km/day between two states using finite differences.

    If `gap` is provided, it's used as the time delta in days; otherwise it's computed
    from the difference in `day` fields.
    """
    lat1, lon1 = eddy_prev.centroid
    lat2, lon2 = eddy_current.centroid
    dt = int(gap) if gap is not None else int((eddy_current.day - eddy_prev.day) / np.timedelta64(1, "D"))
    if dt <= 0:
        eddy_current.velocity = (0.0, 0.0)
        return
    latm = 0.5 * (lat1 + lat2)
    dx = (lon2 - lon1) * 111.0 * np.cos(np.radians(latm))
    dy = (lat2 - lat1) * 111.0
    eddy_current.velocity = (dx / dt, dy / dt)


def predict_position(eddy: EddyState) -> tuple[float, float]:
    """Predict next-day (lat, lon) given current velocity in km/day."""
    vx, vy = eddy.velocity
    lat0, lon0 = eddy.centroid
    coslat = np.cos(np.radians(lat0))
    dlon = vx / (111.0 * coslat if coslat != 0 else np.inf)
    dlat = vy / 111.0
    return (lat0 + dlat, lon0 + dlon)


def matching_cost_inertia(
    eddy_prev: EddyState,
    eddy_current: EddyState,
    *,
    dist_max: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> float:
    """Compute total cost between two eddies (lower is better).

    Components
    ----------
    - Distance to predicted position (normalized by `dist_max`, weighted by `alpha`).
    - Relative diameter change (weighted by `beta`), with extra penalty over 60%.
    - Relative vorticity change (weighted by `gamma`); hard reject if > 1.0.
    - Type mismatch penalty `delta`.
    """
    # distance to inertial prediction
    latp, lonp = predict_position(eddy_prev)
    lat2, lon2 = eddy_current.centroid
    dist = haversine(lonp, latp, lon2, lat2)
    cost_dist = (dist / dist_max) * alpha

    # diameter term
    d1, d2 = eddy_prev.diameter, eddy_current.diameter
    diff_d = abs(d1 - d2) / np.mean([d1, d2]) if (d1 > 0 and d2 > 0) else 1.0
    cost_diam = beta * (diff_d + max(0.0, diff_d - 0.6))

    # vorticity term
    v1, v2 = eddy_prev.vorticity, eddy_current.vorticity
    mv = np.mean([abs(v1), abs(v2)]) if abs(v1) + abs(v2) > 0 else 1e-9
    diff_v = abs(v1 - v2) / mv
    if diff_v > 1.0:
        return 1e6  # hard reject
    cost_vort = gamma * diff_v

    # type term
    cost_type = 0.0 if eddy_prev.eddy_type == eddy_current.eddy_type else delta

    return cost_dist + cost_diam + cost_vort + cost_type


# -----------------------------------------------------------------------------
# Dataset I/O helpers
# -----------------------------------------------------------------------------

def build_detections_dict(ds: xr.Dataset) -> Dict[np.datetime64, List[dict]]:
    """Rebuild per-day detections from the detection Dataset `ds`.

    Accepts either `eccentricity` (EN) or `excentricidad` (ES) variable names.
    """
    times = ds.time.values
    eddies = ds.eddy.values

    if "eccentricity" in ds.variables:
        ecc_name = "eccentricity"
    elif "excentricidad" in ds.variables:
        ecc_name = "excentricidad"
    else:
        raise KeyError("Eccentricity variable not found (expected 'eccentricity' or 'excentricidad').")

    dets: Dict[np.datetime64, List[dict]] = {}
    for i, t in enumerate(times):
        day_list: List[dict] = []
        for j in eddies:
            latv = ds["centroid_lat"].values[i, j]
            lonv = ds["centroid_lon"].values[i, j]
            if np.isfinite(latv) and np.isfinite(lonv):
                day_list.append(
                    {
                        "time": t,
                        "centroid": (float(latv), float(lonv)),
                        "diameter": float(ds["diameter_km"].values[i, j]),
                        "major_axis": float(ds["major_axis_km"].values[i, j]),
                        "minor_axis": float(ds["minor_axis_km"].values[i, j]),
                        "eccentricity": float(ds[ecc_name].values[i, j]),
                        "vorticity": float(ds["vorticity"].values[i, j]),
                        "type": str(ds["type"].values[i, j]),
                    }
                )
        dets[np.datetime64(t)] = day_list
    return dets


def convert_to_states(dets: Dict[np.datetime64, List[dict]]) -> Dict[np.datetime64, List[EddyState]]:
    """Convert the detection dict into lists of :class:`EddyState` grouped by day."""
    states: Dict[np.datetime64, List[EddyState]] = {}
    for day, lst in dets.items():
        states[day] = [
            EddyState(
                day=dd["time"],
                centroid=dd["centroid"],
                diameter=dd["diameter"],
                major_axis=dd["major_axis"],
                minor_axis=dd["minor_axis"],
                eccentricity=dd["eccentricity"],
                vorticity=dd["vorticity"],
                eddy_type=dd["type"],
            )
            for dd in lst
        ]
    return states


# -----------------------------------------------------------------------------
# Tracking driver
# -----------------------------------------------------------------------------

def track_eddies(
    states_dict: Dict[np.datetime64, List[EddyState]],
    *,
    dist_max: float,
    cost_threshold: float,
    allowed_gap: int,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 1.0,
    delta: float = 100.0,
    gap_penalty_per_day: float = 0.1,
) -> List[List[EddyState]]:
    """Greedy inertial-cost matching with gap tolerance.

    Returns a list of tracks; each track is a list of :class:`EddyState` in time order.
    """
    days = sorted(states_dict.keys())
    tracks: List[List[EddyState]] = []
    next_id = 0

    # seed from day 0
    if days:
        for ed in states_dict[days[0]]:
            ed.track_id = next_id
            tracks.append([ed])
            next_id += 1

    # iterate subsequent days
    for day in days[1:]:
        for ed_c in states_dict[day]:
            # candidate tracks within the allowed gap window
            candidates: List[tuple[List[EddyState], EddyState, int]] = []
            for tr in tracks:
                last = tr[-1]
                gap = int((day - last.day) / np.timedelta64(1, "D"))
                if gap <= allowed_gap:
                    candidates.append((tr, last, gap))

            # pick best by cost
            best: Optional[tuple[List[EddyState], EddyState, int]] = None
            best_cost = np.inf
            for tr, last, gap in candidates:
                cost = matching_cost_inertia(
                    last, ed_c,
                    dist_max=dist_max, alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                )
                if gap > 1:
                    cost *= (1.0 + gap_penalty_per_day * (gap - 1))
                if cost < best_cost:
                    best_cost, best = cost, (tr, last, gap)

            if best is not None and best_cost < cost_threshold:
                tr, last, gap = best
                ed_c.track_id = tr[0].track_id
                update_velocity(last, ed_c, gap)
                tr.append(ed_c)
            else:
                ed_c.track_id = next_id
                tracks.append([ed_c])
                next_id += 1

    return tracks


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def tracks_to_dataframe(tracks: List[List[EddyState]]):
    """Convert tracks into a tidy pandas.DataFrame (track_id, time, lat, lon, ...)."""
    import pandas as pd

    rows = []
    for tr_id, elems in enumerate(tracks):
        for st in elems:
            lat, lon = st.centroid
            rows.append(
                {
                    "track_id": tr_id if st.track_id is None else st.track_id,
                    "time": st.day,
                    "lat": float(lat),
                    "lon": float(lon),
                    "diameter_km": float(st.diameter),
                    "major_axis_km": float(st.major_axis),
                    "minor_axis_km": float(st.minor_axis),
                    "eccentricity": float(st.eccentricity),
                    "vorticity": float(st.vorticity),
                    "type": str(st.eddy_type),
                }
            )
    df = pd.DataFrame(rows).sort_values(["track_id", "time"]).reset_index(drop=True)
    return df


def save_tracks_csv_nc(tracks: List[List[EddyState]], csv_path: str, nc_path: str) -> None:
    """Save tracks as CSV and as a NetCDF table for portability."""
    import pandas as pd

    df = tracks_to_dataframe(tracks)
    df.to_csv(csv_path, index=False)

    # store flat table into a Dataset indexed by (track_id, time)
    ds_out = xr.Dataset.from_dataframe(df.set_index(["track_id", "time"]))
    ds_out.to_netcdf(nc_path)


# -----------------------------------------------------------------------------
# Convenience runner (optional)
# -----------------------------------------------------------------------------

def run_from_detection_dataset(ds: xr.Dataset, params: TrackParams):
    """End-to-end: build states from `ds`, track with `params`, return tracks & df.

    Example
    -------
    >>> ds = xr.open_dataset("remolinos_completo_refinado2_2cm.nc")
    >>> params = TrackParams()
    >>> tracks, df = run_from_detection_dataset(ds, params)
    >>> df.to_csv("tracks_refined.csv", index=False)
    """
    dets = build_detections_dict(ds)
    states = convert_to_states(dets)
    tracks = track_eddies(
        states,
        dist_max=params.dist_max_km,
        cost_threshold=params.cost_threshold,
        allowed_gap=params.allowed_gap_days,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
        delta=params.delta,
        gap_penalty_per_day=params.gap_penalty_per_day,
    )
    df = tracks_to_dataframe(tracks)
    return tracks, df
