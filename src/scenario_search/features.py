"""Scenario measures computed from ego motion, CAN bus, HD map and 3D box annotations.

``compute_measures`` turns one :class:`ScenarioLog` into a flat dict of numbers
(e.g. ``turn_left_deg``, ``n_peds_crossing``) plus evidence time spans. Turning
measures into tag scores is done in :mod:`scenario_search.tags`, so thresholds can
be tuned without recomputing geometry.
"""

from __future__ import annotations

import numpy as np

from .geometry import to_ego_frame, unwrap_yaw, wrap_angle
from .hdmap import HDMap
from .nuscenes_io import ATTRIBUTES, GROUPS, ScenarioLog
from .solar import scene_sun_elevation

G = {g: i for i, g in enumerate(GROUPS)}
A = {a: i for i, a in enumerate(ATTRIBUTES)}

PED_GROUPS = [
    G[g] for g in ("pedestrian", "child", "police_officer", "construction_worker", "stroller", "wheelchair", "scooter")
]
VEHICLE_GROUPS = [
    G[g] for g in ("car", "truck", "bus", "motorcycle", "trailer", "construction_vehicle", "emergency_vehicle")
]
NEAR_M = 30.0
EGO_HZ = 4.0  # resample ego trajectory to 4 Hz for map queries


# --------------------------------------------------------------------------- ego
def ego_track(slog: ScenarioLog, hz: float = EGO_HZ) -> dict[str, np.ndarray]:
    """Resampled ego trajectory with smoothed speed / accel / yaw rate."""
    t_raw = slog.ego_t
    t = np.arange(t_raw[0], t_raw[-1] + 1e-6, 1.0 / hz)
    x = np.interp(t, t_raw, slog.ego_xy[:, 0])
    y = np.interp(t, t_raw, slog.ego_xy[:, 1])
    yaw = np.interp(t, t_raw, unwrap_yaw(slog.ego_yaw))
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    speed = np.hypot(dx, dy)
    k = max(1, int(hz / 2))
    speed = np.convolve(np.pad(speed, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    # Prefer CAN bus speed when present (wheel odometry, km/h).
    if "vm_t" in slog.can and len(slog.can["vm_t"]) > 3:
        speed = np.interp(t, slog.can["vm_t"], slog.can["vehicle_speed"] / 3.6)
    accel = np.gradient(speed, t)
    yaw_rate = np.gradient(yaw, t)
    return {"t": t, "xy": np.stack([x, y], 1), "yaw": yaw, "speed": speed, "accel": accel, "yaw_rate": yaw_rate}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def turn_measures(tr: dict, max_window_s: float = 12.0) -> dict:
    """Largest signed heading change inside any window while the ego is moving."""
    t, yaw, _speed = tr["t"], tr["yaw"], tr["speed"]
    dist = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(tr["xy"], axis=0).T))])
    best = {"left": (0.0, 0, 0), "right": (0.0, 0, 0)}
    n = len(t)
    w = int(max_window_s * EGO_HZ)
    for i in range(n):
        j_end = min(n, i + w + 1)
        dyaw = np.degrees(yaw[i:j_end] - yaw[i])
        moved = dist[i:j_end] - dist[i]
        ok = moved > 4.0
        if not ok.any():
            continue
        jl = int(np.argmax(np.where(ok, dyaw, -np.inf)))
        jr = int(np.argmin(np.where(ok, dyaw, np.inf)))
        if dyaw[jl] > best["left"][0]:
            best["left"] = (float(dyaw[jl]), i, i + jl)
        if -dyaw[jr] > best["right"][0]:
            best["right"] = (float(-dyaw[jr]), i, i + jr)
    out = {
        "turn_left_deg": best["left"][0],
        "turn_right_deg": best["right"][0],
        "net_heading_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "max_abs_heading_deg": float(np.degrees(np.max(np.abs(yaw - yaw[0])))),
    }
    spans = {
        "turn_left": (float(t[best["left"][1]]), float(t[best["left"][2]])),
        "turn_right": (float(t[best["right"][1]]), float(t[best["right"][2]])),
    }
    return out | {"_spans": spans}


def speed_measures(tr: dict, slog: ScenarioLog) -> dict:
    t, speed, accel = tr["t"], tr["speed"], tr["accel"]
    stopped = speed < 0.3
    runs = _runs(stopped)
    stop_runs = [(a, b) for a, b in runs if (b - a) / EGO_HZ >= 1.5]
    # A "stop" is coming to a halt after moving; a "start" is moving off after a halt.
    came_to_stop = [(a, b) for a, b in stop_runs if a > 0 and speed[:a].max() > 2.0]
    moved_off = [(a, b) for a, b in stop_runs if b < len(t) and speed[b:].max() > 2.0]
    if "pose_accel" in slog.can and len(slog.can["pose_accel"]) > 5:
        a_can = slog.can["pose_accel"]
        min_accel = float(np.percentile(a_can, 1))
    else:
        min_accel = float(np.min(accel))
    out = {
        "max_speed_kmh": float(speed.max() * 3.6),
        "mean_speed_kmh": float(speed.mean() * 3.6),
        "frac_stopped": float(stopped.mean()),
        "longest_stop_s": float(max([(b - a) / EGO_HZ for a, b in runs], default=0.0)),
        "n_stops": float(len(came_to_stop)),
        "n_starts": float(len(moved_off)),
        "min_accel_mps2": min_accel,
        "distance_m": float(np.sum(np.hypot(*np.diff(tr["xy"], axis=0).T))),
    }
    reverse = 0.0
    if "gear_position" in slog.can and len(slog.can["gear_position"]):
        # nuScenes Renault Zoe: gear 0 = park, 1 = reverse (README of CAN bus expansion).
        moving = slog.can["vehicle_speed"] > 1.0
        reverse = float(np.mean((slog.can["gear_position"] == 1) & moving))
    out["frac_reverse"] = reverse
    spans = {}
    if came_to_stop:
        a, b = max(came_to_stop, key=lambda r: r[1] - r[0])
        spans["stop"] = (float(t[a]), float(t[b - 1]))
    if moved_off:
        a, b = moved_off[0]
        spans["start_from_stop"] = (float(t[a]), float(t[min(b, len(t) - 1)]))
    return out | {"_spans": spans}


def signal_measures(slog: ScenarioLog) -> dict:
    c = slog.can
    if "left_signal" not in c:
        return {"left_signal_frac": 0.0, "right_signal_frac": 0.0, "has_can": 0.0}
    return {
        "left_signal_frac": float(np.mean(c["left_signal"] > 0)),
        "right_signal_frac": float(np.mean(c["right_signal"] > 0)),
        "has_can": 1.0,
    }


# --------------------------------------------------------------------------- map
def map_measures(tr: dict, hdmap: HDMap) -> dict:
    xy, t, yaw = tr["xy"], tr["t"], tr["yaw"]
    ahead = xy + 10.0 * np.stack([np.cos(yaw), np.sin(yaw)], 1)
    in_int = hdmap.in_intersection(xy)
    near_int = in_int | hdmap.in_intersection(ahead)
    on_xwalk = hdmap.any_contains("ped_crossing", xy)
    in_park = hdmap.any_contains("carpark_area", xy)
    stop_types = hdmap.stop_lines_near(xy, radius=4.0)
    stopped = tr["speed"] < 0.3

    near_park = hdmap.near("carpark_area", xy, 10.0)
    wait_int = stopped & near_int
    wait_runs = [r for r in _runs(wait_int) if (r[1] - r[0]) / EGO_HZ >= 2.0]
    passed_through = bool(in_int.any() and tr["speed"][in_int].max() > 2.0)

    out = {
        "frac_in_intersection": float(in_int.mean()),
        "enters_intersection": float(in_int.any()),
        "crosses_intersection": float(passed_through),
        "frac_near_intersection": float(near_int.mean()),
        "crosswalks_crossed": float(len(_runs(on_xwalk))),
        "frac_in_carpark": float(in_park.mean()),
        "frac_near_carpark": float(near_park.mean()),
        "longest_wait_near_intersection_s": float(max([(b - a) / EGO_HZ for a, b in _runs(wait_int)], default=0.0)),
        "near_stop_sign": float(any("STOP_SIGN" in s for s in stop_types)),
        "near_traffic_light": float(any("TRAF_LIGHT" in s for s in stop_types)),
        "wait_at_intersection_s": float(sum((b - a) / EGO_HZ for a, b in wait_runs)),
    }
    spans = {}
    if in_int.any():
        idx = np.where(in_int)[0]
        spans["intersection"] = (float(t[idx[0]]), float(t[idx[-1]]))
    if wait_runs:
        a, b = wait_runs[0]
        spans["wait_at_intersection"] = (float(t[a]), float(t[b - 1]))

    # Lane changes: consecutive (non-connector) lanes that are not connected in the lane graph.
    lanes = hdmap.lane_of(xy)
    seq = [(i, ln) for i, ln in enumerate(lanes) if ln is not None]
    runs: list[list] = []
    for i, ln in seq:
        if runs and runs[-1][0] == ln:
            runs[-1][2] = i
        else:
            runs.append([ln, i, i])
    n_lc, lc_left, lc_right, lc_span = 0, 0, 0, None
    for prev, cur in zip(runs, runs[1:]):
        if (cur[2] - cur[1] + 1) / EGO_HZ < 1.0 or (prev[2] - prev[1] + 1) / EGO_HZ < 1.0:
            continue
        if cur[0] in hdmap.lanes_reachable(prev[0], hops=3):
            continue
        i0, i1 = prev[2], cur[1]
        lo, hi = max(0, i0 - 8), min(len(t) - 1, i1 + 8)
        dyaw = float(np.degrees(np.ptp(yaw[lo : hi + 1])))
        if dyaw > 35 or in_int[lo : hi + 1].any():
            continue
        lateral = to_ego_frame(xy[hi : hi + 1], xy[lo], yaw[lo])[0, 1]
        n_lc += 1
        if lateral > 0:
            lc_left += 1
        else:
            lc_right += 1
        lc_span = lc_span or (float(t[lo]), float(t[hi]))
    # Geometric lane-change detector: ~one lane width of lateral shift with little net
    # heading change, outside intersections. Complements the lane-graph detector above.
    n_geo, geo_span = 0, None
    w = int(7 * EGO_HZ)
    i = 0
    while i + 4 < len(t):
        j = min(len(t) - 1, i + w)
        seg = slice(i, j + 1)
        if tr["speed"][seg].min() < 2.0 or in_int[seg].any():
            i += 2
            continue
        local = to_ego_frame(xy[seg], xy[i], yaw[i])
        dyaw_end = abs(np.degrees(yaw[j] - yaw[i]))
        lat = (
            local[-1, 1] - np.polyval(np.polyfit(local[:3, 0], local[:3, 1], 1), local[-1, 0])
            if np.ptp(local[:3, 0]) > 0.5
            else local[-1, 1]
        )
        if dyaw_end < 8 and 2.4 < abs(lat) < 5.0 and np.degrees(np.ptp(yaw[seg])) < 25:
            n_geo += 1
            geo_span = geo_span or (float(t[i]), float(t[j]))
            i = j
        else:
            i += 2
    if n_geo and not lc_span:
        lc_span = geo_span
    out["n_lane_changes_geo"] = float(n_geo)
    out |= {
        "n_lane_changes": float(n_lc),
        "n_lane_changes_left": float(lc_left),
        "n_lane_changes_right": float(lc_right),
    }
    if lc_span:
        spans["lane_change"] = lc_span
    return out | {"_spans": spans, "_in_int": in_int}


# ------------------------------------------------------------------------ agents
def agent_measures(slog: ScenarioLog, tr: dict, hdmap: HDMap) -> dict:
    """Counts and interactions of annotated agents around the ego vehicle."""
    st = slog.sample_t
    ex = np.interp(st, tr["t"], tr["xy"][:, 0])
    ey = np.interp(st, tr["t"], tr["xy"][:, 1])
    eyaw = np.interp(st, tr["t"], tr["yaw"])
    espeed = np.interp(st, tr["t"], tr["speed"])
    s = slog.ann_sample.astype(int)
    d = slog.ann_xy - np.stack([ex[s], ey[s]], 1)
    c, si = np.cos(-eyaw[s]), np.sin(-eyaw[s])
    rel = np.stack([c * d[:, 0] - si * d[:, 1], si * d[:, 0] + c * d[:, 1]], 1)
    dist = np.hypot(rel[:, 0], rel[:, 1])
    near = dist < NEAR_M
    grp, attr, inst = slog.ann_group, slog.ann_attr, slog.ann_inst
    is_ped = np.isin(grp, PED_GROUPS)
    is_veh = np.isin(grp, VEHICLE_GROUPS)

    def uniq(mask) -> float:
        return float(len(np.unique(inst[mask])))

    out: dict[str, float] = {}
    for g, gi in G.items():
        out[f"n_{g}"] = uniq(near & (grp == gi))
    # Objects a viewer of the front camera would notice: inside the ~70 deg CAM_FRONT
    # field of view, within 40 m, and at least 40% visible.
    bearing = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    front_vis = (rel[:, 0] > 1) & (np.abs(bearing) < 35) & (dist < 40) & (slog.ann_vis >= 2)
    for g, gi in G.items():
        out[f"front_{g}"] = uniq(front_vis & (grp == gi))
    out["front_peds"] = uniq(front_vis & is_ped)
    out["front_parked_vehicles"] = uniq(front_vis & is_veh & (attr == A["vehicle.parked"]))
    out["front_cyclists"] = uniq(front_vis & (grp == G["bicycle"]) & (attr == A["cycle.with_rider"]))
    out["front_parked_bicycles"] = uniq(front_vis & (grp == G["bicycle"]) & (attr == A["cycle.without_rider"]))
    out["front_motorcyclists"] = uniq(front_vis & (grp == G["motorcycle"]) & (attr == A["cycle.with_rider"]))
    out["front_traffic_vehicles"] = uniq(front_vis & is_veh & (attr == A["vehicle.moving"]))
    out["n_peds"] = uniq(near & is_ped)
    out["n_peds_moving"] = uniq(near & is_ped & (attr == A["pedestrian.moving"]))
    out["n_peds_sitting"] = uniq(near & is_ped & (attr == A["pedestrian.sitting_lying_down"]))
    out["n_cyclists"] = uniq(near & (grp == G["bicycle"]) & (attr == A["cycle.with_rider"]))
    out["n_parked_bicycles"] = uniq(near & (grp == G["bicycle"]) & (attr == A["cycle.without_rider"]))
    out["n_motorcyclists"] = uniq(near & (grp == G["motorcycle"]) & (attr == A["cycle.with_rider"]))
    out["n_parked_motorcycles"] = uniq(near & (grp == G["motorcycle"]) & (attr == A["cycle.without_rider"]))
    out["n_parked_vehicles"] = uniq(near & is_veh & (attr == A["vehicle.parked"]))
    out["n_vehicles"] = uniq(near & is_veh)
    traffic = near & is_veh & np.isin(attr, [A["vehicle.moving"], A["vehicle.stopped"]])
    per_frame = np.bincount(s[traffic], minlength=len(st))
    out["mean_traffic_vehicles"] = float(per_frame.mean())
    out["max_traffic_vehicles"] = float(per_frame.max(initial=0))
    out["mean_ego_speed_kmh_at_frames"] = float(espeed.mean() * 3.6)
    per_frame_peds = np.bincount(s[near & is_ped], minlength=len(st))
    out["max_peds_in_frame"] = float(per_frame_peds.max(initial=0))

    # Pedestrians on the road surface in front of the ego vehicle.
    front = (rel[:, 0] > -2) & (rel[:, 0] < 30) & (np.abs(rel[:, 1]) < 12)
    ped_rows = np.where(is_ped & front & (attr == A["pedestrian.moving"]))[0]
    spans = {}
    if len(ped_rows):
        on_road = hdmap.on_road(slog.ann_xy[ped_rows])
        on_xwalk = hdmap.any_contains("ped_crossing", slog.ann_xy[ped_rows])
        crossing = ped_rows[on_road | on_xwalk]
        jay = ped_rows[on_road & ~on_xwalk & ~hdmap.in_intersection(slog.ann_xy[ped_rows])]
        out["n_peds_crossing"] = uniq(np.isin(np.arange(len(inst)), crossing))
        out["n_jaywalkers"] = uniq(np.isin(np.arange(len(inst)), jay))
        if len(crossing):
            ts = st[s[crossing]]
            spans["ped_crossing"] = (float(ts.min()), float(ts.max()))
        if len(jay):
            ts = st[s[jay]]
            spans["jaywalker"] = (float(ts.min()), float(ts.max()))
    else:
        out["n_peds_crossing"] = 0.0
        out["n_jaywalkers"] = 0.0

    # Vehicle interactions via per-instance tracks in the ego frame.
    lead_frames = np.zeros(len(st), dtype=bool)
    n_cut_in, n_overtaken, cut_span = 0, 0, None
    veh_rows = np.where(is_veh & (dist < 45))[0]
    for iid in np.unique(inst[veh_rows]):
        rows = veh_rows[inst[veh_rows] == iid]
        rows = rows[np.argsort(s[rows])]
        fr, xs, ys = s[rows], rel[rows, 0], rel[rows, 1]
        moving = attr[rows] == A["vehicle.moving"]
        lead = (xs > 2) & (xs < 30) & (np.abs(ys) < 1.8)
        lead_frames[fr[lead]] = True
        for k in range(len(fr) - 1):
            # cut-in: lateral offset drops from an adjacent lane into ego lane ahead
            if moving[k] and abs(ys[k]) > 2.6 and xs[k] > 0:
                later = (fr > fr[k]) & (fr <= fr[k] + 6) & (np.abs(ys) < 1.2) & (xs > 2) & (xs < 25)
                if later.any():
                    n_cut_in += 1
                    cut_span = cut_span or (float(st[fr[k]]), float(st[fr[later][0]]))
                    break
        behind = (xs < -3) & (np.abs(ys) < 6)
        ahead_ = (xs > 3) & (np.abs(ys) < 6)
        if behind.any() and ahead_.any() and fr[np.argmax(behind)] < fr[ahead_][-1] and moving.any():
            n_overtaken += 1
    ego_moving_frames = espeed > 2.0
    out["frac_following"] = float(lead_frames[ego_moving_frames].mean()) if ego_moving_frames.any() else 0.0
    out["n_cut_ins"] = float(n_cut_in)
    out["n_overtaken_by"] = float(n_overtaken)
    if cut_span:
        spans["cut_in"] = cut_span

    # Construction zone evidence.
    out["construction_score_raw"] = (
        out["n_cone"] + out["n_barrier"] + 3 * out["n_construction_worker"] + 3 * out["n_construction_vehicle"]
    )
    return out | {"_spans": spans}


def oncoming_during_turn(slog: ScenarioLog, tr: dict, span: tuple[float, float]) -> float:
    """Number of moving vehicles heading roughly against the ego's pre-turn heading."""
    t0, t1 = span
    if t1 <= t0:
        return 0.0
    st = slog.sample_t
    frames = np.where((st >= t0 - 1) & (st <= t1))[0]
    if not len(frames):
        return 0.0
    yaw0 = float(np.interp(t0, tr["t"], tr["yaw"]))
    x0 = np.array([np.interp(t0, tr["t"], tr["xy"][:, 0]), np.interp(t0, tr["t"], tr["xy"][:, 1])])
    rows = np.where(
        np.isin(slog.ann_sample, frames)
        & np.isin(slog.ann_group, VEHICLE_GROUPS)
        & (slog.ann_attr == A["vehicle.moving"])
    )[0]
    if not len(rows):
        return 0.0
    rel = to_ego_frame(slog.ann_xy[rows], x0, yaw0)
    dyaw = np.abs(wrap_angle(slog.ann_yaw[rows] - yaw0))
    m = (dyaw > np.radians(135)) & (rel[:, 0] > 0) & (rel[:, 0] < 50) & (np.abs(rel[:, 1]) < 15)
    return float(len(np.unique(slog.ann_inst[rows][m])))


# ------------------------------------------------------------------------ public
def compute_measures(slog: ScenarioLog, hdmap: HDMap) -> dict:
    tr = ego_track(slog)
    parts = [
        turn_measures(tr),
        speed_measures(tr, slog),
        signal_measures(slog),
        map_measures(tr, hdmap),
        agent_measures(slog, tr, hdmap),
    ]
    measures: dict = {}
    spans: dict = {}
    for p in parts:
        spans.update(p.pop("_spans", {}))
        p.pop("_in_int", None)
        measures.update(p)

    # Turns across oncoming traffic: left in right-hand traffic (Boston), right in Singapore.
    lht = slog.location.startswith("singapore")
    across = "turn_right" if lht else "turn_left"
    measures["turn_across_traffic_deg"] = measures[f"{across}_deg"]
    measures["oncoming_during_turn"] = oncoming_during_turn(slog, tr, spans.get(across, (0, 0)))
    if measures["turn_across_traffic_deg"] > 0:
        spans["turn_across_traffic"] = spans.get(across, (0, 0))

    # Lane change evidence: lane-graph detector, geometric detector, and a turn signal
    # used while the heading barely changes (a signalled lane change rather than a turn).
    signal = max(measures["left_signal_frac"], measures["right_signal_frac"])
    measures["lane_change_evidence"] = (
        float(measures["n_lane_changes"] > 0)
        + float(measures["n_lane_changes_geo"] > 0)
        + float(signal > 0.1 and measures["max_abs_heading_deg"] < 30)
    )

    sun = scene_sun_elevation(slog.start_utime, slog.location)
    measures["sun_elevation_deg"] = sun
    measures["is_singapore"] = float(lht)
    measures["is_boston"] = float(slog.location == "boston-seaport")
    return {
        "measures": measures,
        "spans": spans,
        "track": {
            "t": tr["t"][::2].round(2).tolist(),
            "xy": to_ego_frame(tr["xy"][::2], tr["xy"][0], tr["yaw"][0]).round(2).tolist(),
            "speed_kmh": (tr["speed"][::2] * 3.6).round(1).tolist(),
        },
    }
