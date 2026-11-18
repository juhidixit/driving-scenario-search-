"""Export a logged scenario as an ASAM OpenSCENARIO 1.2 (.xosc) replay.

The ego vehicle and every annotated vehicle / pedestrian / cyclist that is tracked for
at least three key frames become entities that follow their logged trajectories
(``FollowTrajectoryAction`` with timed polyline vertices). This is a log-replay
scenario: the typical starting point for turning a mined real-world event into a
parameterised simulation test.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np

from .features import VEHICLE_GROUPS, ego_track
from .nuscenes_io import GROUPS, ScenarioLog

PED_LIKE = {"pedestrian", "child", "police_officer", "construction_worker", "stroller", "wheelchair", "scooter"}
CYCLE_LIKE = {"bicycle", "motorcycle"}
VEHICLE_CATEGORY = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "construction_vehicle": "truck",
    "emergency_vehicle": "car",
    "motorcycle": "motorbike",
    "bicycle": "bicycle",
}


def _sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _bbox(parent, w, length, h):
    bb = _sub(parent, "BoundingBox")
    _sub(bb, "Center", x=0, y=0, z=round(h / 2, 3))
    _sub(bb, "Dimensions", width=round(w, 3), length=round(length, 3), height=round(h, 3))


def _entity(entities, name: str, group: str, size) -> None:
    so = _sub(entities, "ScenarioObject", name=name)
    w, length, h = (float(s) for s in size)
    if group in PED_LIKE:
        p = _sub(so, "Pedestrian", name=name, mass=80, pedestrianCategory="pedestrian", model3d="")
        _sub(p, "ParameterDeclarations")
        _bbox(p, w, length, h)
        _sub(p, "Properties")
        return
    v = _sub(so, "Vehicle", name=name, vehicleCategory=VEHICLE_CATEGORY.get(group, "car"))
    _sub(v, "ParameterDeclarations")
    _bbox(v, w, length, h)
    perf = _sub(v, "Performance", maxSpeed=70, maxAcceleration=10, maxDeceleration=10)
    perf.tail = None
    axles = _sub(v, "Axles")
    _sub(
        axles,
        "FrontAxle",
        maxSteering=0.5,
        wheelDiameter=0.6,
        trackWidth=1.6,
        positionX=round(length * 0.35, 2),
        positionZ=0.3,
    )
    _sub(
        axles,
        "RearAxle",
        maxSteering=0,
        wheelDiameter=0.6,
        trackWidth=1.6,
        positionX=round(-length * 0.35, 2),
        positionZ=0.3,
    )
    _sub(v, "Properties")


def _world(parent, x, y, h):
    pos = _sub(parent, "Position")
    _sub(pos, "WorldPosition", x=round(float(x), 3), y=round(float(y), 3), z=0, h=round(float(h), 4))
    return pos


def _trajectory_story(acts_parent, actor: str, t, xy, yaw, idx: int) -> None:
    mg = _sub(acts_parent, "ManeuverGroup", maximumExecutionCount=1, name=f"mg_{actor}")
    actors = _sub(mg, "Actors", selectTriggeringEntities="false")
    _sub(actors, "EntityRef", entityRef=actor)
    man = _sub(mg, "Maneuver", name=f"replay_{actor}")
    ev = _sub(man, "Event", name=f"ev_{actor}", priority="override")
    act = _sub(ev, "Action", name=f"follow_{actor}")
    pa = _sub(act, "PrivateAction")
    ra = _sub(pa, "RoutingAction")
    fta = _sub(ra, "FollowTrajectoryAction")
    traj = _sub(fta, "Trajectory", name=f"log_{actor}", closed="false")
    _sub(traj, "ParameterDeclarations")
    shape = _sub(traj, "Shape")
    poly = _sub(shape, "Polyline")
    for ti, (x, y), h in zip(t, xy, yaw):
        vert = _sub(poly, "Vertex", time=round(float(ti), 3))
        _world(vert, x, y, h)
    tr = _sub(fta, "TimeReference")
    _sub(tr, "Timing", domainAbsoluteRelative="absolute", scale=1, offset=0)
    _sub(fta, "TrajectoryFollowingMode", followingMode="position")
    st = _sub(ev, "StartTrigger")
    cg = _sub(st, "ConditionGroup")
    cond = _sub(cg, "Condition", name=f"start_{actor}", delay=0, conditionEdge="rising")
    bv = _sub(cond, "ByValueCondition")
    _sub(bv, "SimulationTimeCondition", value=0, rule="greaterThan")


def to_xosc(slog: ScenarioLog, max_agents: int = 60, min_frames: int = 3) -> str:
    tr = ego_track(slog)
    origin = tr["xy"][0]
    root = ET.Element("OpenSCENARIO")
    _sub(
        root,
        "FileHeader",
        revMajor=1,
        revMinor=2,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        description=f"nuScenes {slog.name} ({slog.location}) log replay",
        author="driving-scenario-search",
    )
    _sub(root, "ParameterDeclarations")
    _sub(root, "CatalogLocations")
    rn = _sub(root, "RoadNetwork")
    _sub(rn, "LogicFile", filepath=f"{slog.location}.xodr")  # placeholder: no OpenDRIVE export
    entities = _sub(root, "Entities")
    _entity(entities, "Ego", "car", (1.73, 4.08, 1.56))  # Renault Zoe

    # Agents: keep tracks with enough frames, nearest first.
    agents = []
    for iid in np.unique(slog.ann_inst):
        rows = np.where(slog.ann_inst == iid)[0]
        g = GROUPS[slog.ann_group[rows[0]]]
        if len(rows) < min_frames or not (
            g in PED_LIKE or g in CYCLE_LIKE or slog.ann_group[rows[0]] in VEHICLE_GROUPS
        ):
            continue
        rows = rows[np.argsort(slog.ann_sample[rows])]
        d = np.hypot(*(slog.ann_xy[rows].mean(0) - origin))
        agents.append((d, iid, g, rows))
    agents = sorted(agents, key=lambda a: a[0])[:max_agents]
    for k, (_, _, g, rows) in enumerate(agents):
        _entity(entities, f"{g}_{k}", g, slog.ann_size[rows[0]])

    sb = _sub(root, "Storyboard")
    init = _sub(sb, "Init")
    ia = _sub(init, "Actions")

    def teleport(name, x, y, h):
        pr = _sub(ia, "Private", entityRef=name)
        pa = _sub(pr, "PrivateAction")
        ta = _sub(pa, "TeleportAction")
        _world(ta, x - origin[0], y - origin[1], h)

    step = max(1, len(tr["t"]) // 80)
    teleport("Ego", *tr["xy"][0], tr["yaw"][0])
    for k, (_, _, g, rows) in enumerate(agents):
        teleport(f"{g}_{k}", *slog.ann_xy[rows[0]], slog.ann_yaw[rows[0]])

    story = _sub(sb, "Story", name=f"replay_{slog.name}")
    act = _sub(story, "Act", name="replay")
    _trajectory_story(act, "Ego", tr["t"][::step], tr["xy"][::step] - origin, tr["yaw"][::step], 0)
    for k, (_, _, g, rows) in enumerate(agents):
        t = slog.sample_t[slog.ann_sample[rows]]
        _trajectory_story(act, f"{g}_{k}", t, slog.ann_xy[rows] - origin, slog.ann_yaw[rows], k + 1)
    st = _sub(act, "StartTrigger")
    cg = _sub(st, "ConditionGroup")
    c = _sub(cg, "Condition", name="act_start", delay=0, conditionEdge="rising")
    bv = _sub(c, "ByValueCondition")
    _sub(bv, "SimulationTimeCondition", value=0, rule="greaterThan")
    stop = _sub(sb, "StopTrigger")
    cg = _sub(stop, "ConditionGroup")
    c = _sub(cg, "Condition", name="end", delay=0, conditionEdge="rising")
    bv = _sub(c, "ByValueCondition")
    _sub(bv, "SimulationTimeCondition", value=round(float(tr["t"][-1]) + 0.5, 2), rule="greaterThan")

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
