"""Synthetic fixtures: a tiny HD map and hand-built scenario logs (no nuScenes download needed)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scenario_search.nuscenes_io import ATTRIBUTES, GROUPS, ScenarioLog

LOCATION = "boston-seaport"
T0 = 1_538_000_000_000_000  # 2018-09-26 ~22:13 UTC -> Boston evening


def _poly(token, pts, nodes, polys):
    toks = []
    for i, (x, y) in enumerate(pts):
        nt = f"{token}_n{i}"
        nodes.append({"token": nt, "x": x, "y": y})
        toks.append(nt)
    polys.append({"token": f"{token}_p", "exterior_node_tokens": toks, "holes": []})
    return f"{token}_p"


@pytest.fixture(scope="session")
def map_dir(tmp_path_factory):
    """Straight road along +x (two lanes), a 4-way intersection at x in [40, 60], a crosswalk and a car park."""
    nodes, polys = [], []
    road = {
        "road_segment": [
            {
                "token": "rs_int",
                "polygon_token": _poly("rs_int", [(40, -10), (60, -10), (60, 10), (40, 10)], nodes, polys),
                "is_intersection": True,
                "drivable_area_token": "",
            },
            {
                "token": "rs_a",
                "polygon_token": _poly("rs_a", [(-100, -4), (40, -4), (40, 4), (-100, 4)], nodes, polys),
                "is_intersection": False,
                "drivable_area_token": "",
            },
        ],
        "lane": [
            {
                "token": "lane_r",
                "polygon_token": _poly("lane_r", [(-100, -4), (40, -4), (40, 0), (-100, 0)], nodes, polys),
                "lane_type": "CAR",
            },
            {
                "token": "lane_l",
                "polygon_token": _poly("lane_l", [(-100, 0), (40, 0), (40, 4), (-100, 4)], nodes, polys),
                "lane_type": "CAR",
            },
        ],
        "lane_connector": [],
        "ped_crossing": [
            {"token": "pc", "polygon_token": _poly("pc", [(34, -6), (38, -6), (38, 6), (34, 6)], nodes, polys)}
        ],
        "walkway": [],
        "carpark_area": [
            {"token": "cp", "polygon_token": _poly("cp", [(-100, 6), (0, 6), (0, 30), (-100, 30)], nodes, polys)}
        ],
        "stop_line": [
            {
                "token": "sl",
                "polygon_token": _poly("sl", [(37, -4), (39, -4), (39, 0), (37, 0)], nodes, polys),
                "stop_line_type": "TRAF_LIGHT",
            }
        ],
    }
    data = {"version": "1.3", "node": nodes, "polygon": polys, "connectivity": {}, **road}
    d = tmp_path_factory.mktemp("maps")
    (d / f"{LOCATION}.json").write_text(json.dumps(data))
    return d


def make_log(name: str, xy: np.ndarray, yaw: np.ndarray, agents=(), t0: int = T0, dt: float = 0.05) -> ScenarioLog:
    """agents: list of (group, attribute, [(sample_idx, x, y, yaw), ...])."""
    ego_t = np.arange(len(xy)) * dt
    sample_t = np.arange(0, ego_t[-1] + 1e-9, 0.5)
    rows = []
    for iid, (group, attr, track) in enumerate(agents):
        for si, x, y, h in track:
            rows.append((si, iid, GROUPS.index(group), ATTRIBUTES.index(attr), x, y, h))
    a = np.array(rows, dtype=float) if rows else np.zeros((0, 7))
    return ScenarioLog(
        name=name,
        token=name,
        location=LOCATION,
        log_name="log-" + name,
        date="2018-09-26",
        start_utime=t0,
        description="",
        sample_t=sample_t,
        cam_front=[""] * len(sample_t),
        ego_t=ego_t,
        ego_xy=np.asarray(xy, float),
        ego_yaw=np.asarray(yaw, float),
        ann_sample=a[:, 0].astype(np.int16),
        ann_inst=a[:, 1].astype(np.int32),
        ann_group=a[:, 2].astype(np.int8),
        ann_attr=a[:, 3].astype(np.int8),
        ann_xy=a[:, 4:6].astype(np.float32),
        ann_yaw=a[:, 6].astype(np.float32),
        ann_size=np.tile(np.array([[0.7, 0.7, 1.7]], np.float32), (len(a), 1)),
        ann_vis=np.full(len(a), 4, np.int8),
        can={},
    )


def straight(n=400, speed=10.0, y=-2.0, x0=-80.0, dt=0.05):
    x = x0 + speed * dt * np.arange(n)
    return np.stack([x, np.full(n, y)], 1), np.zeros(n)


def left_turn(n=400, speed=6.0, dt=0.05):
    """Drive +x to the intersection centre, then a 90 deg left arc, then +y."""
    pts, yaws = [], []
    x, y, h = 20.0, -2.0, 0.0
    for _i in range(n):
        if 45 <= x and h < math.pi / 2 and y < 8:
            h = min(math.pi / 2, h + speed * dt / 10.0)  # radius 10 m
        x += speed * dt * math.cos(h)
        y += speed * dt * math.sin(h)
        pts.append((x, y))
        yaws.append(h)
    return np.array(pts), np.array(yaws)


def stop_at_light(n=400, dt=0.05):
    """Approach the stop line, wait 8 s, then drive through."""
    x, pts = 0.0, []
    for i in range(n):
        t = i * dt
        v = 8.5 if t < 4 else 0.0 if t < 12 else 6.0
        x = min(x + v * dt, 36.0) if t < 12 else x + v * dt
        pts.append((x, -2.0))
    return np.array(pts), np.zeros(n)


@pytest.fixture(scope="session")
def logs():
    xy_s, yaw_s = straight()
    peds = [
        ("pedestrian", "pedestrian.moving", [(si, -60 + 5 * si, -6 + 0.6 * si, 1.57) for si in range(20)])
        for _ in range(1)
    ]
    crowd = [
        ("pedestrian", "pedestrian.standing", [(si, -70 + 5 * si + k, 8.0, 0.0) for si in range(0, 40, 4)])
        for k in range(25)
    ]
    out = {
        "scene-straight": make_log("scene-straight", xy_s, yaw_s),
        "scene-left": make_log("scene-left", *left_turn()),
        "scene-stop": make_log("scene-stop", *stop_at_light(), agents=peds),
        "scene-crowd": make_log("scene-crowd", xy_s, yaw_s, agents=crowd),
    }
    return out
