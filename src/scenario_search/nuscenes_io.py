"""Turn raw nuScenes tables into compact per-scene logs.

Only the public nuScenes JSON tables, the CAN bus expansion and CAM_FRONT key frames
are read. The scene ``description`` field (written by human annotators) is copied
into the log for display and for the offline evaluation, but no indexing or search
code reads it (enforced by ``tests/test_no_leakage.py``).
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import quat_to_yaw

log = logging.getLogger(__name__)

CATEGORY_GROUPS = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "child",
    "human.pedestrian.wheelchair": "wheelchair",
    "human.pedestrian.stroller": "stroller",
    "human.pedestrian.personal_mobility": "scooter",
    "human.pedestrian.police_officer": "police_officer",
    "human.pedestrian.construction_worker": "construction_worker",
    "animal": "animal",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "emergency_vehicle",
    "vehicle.emergency.police": "emergency_vehicle",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "cone",
    "movable_object.pushable_pullable": "pushable",
    "movable_object.debris": "debris",
    "static_object.bicycle_rack": "bicycle_rack",
}
GROUPS = sorted(set(CATEGORY_GROUPS.values()))
ATTRIBUTES = [
    "none",
    "vehicle.moving",
    "vehicle.stopped",
    "vehicle.parked",
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.sitting_lying_down",
    "pedestrian.standing",
    "pedestrian.moving",
]


@dataclass
class ScenarioLog:
    """Everything the feature extractors need for one 20 s scene."""

    name: str
    token: str
    location: str
    log_name: str
    date: str
    start_utime: int
    description: str  # human text: display + evaluation only
    sample_t: np.ndarray  # (S,) seconds since start, 2 Hz key frames
    cam_front: list[str]  # (S,) relative jpg path per key frame ("" if missing)
    ego_t: np.ndarray  # (E,) ~20 Hz lidar ego poses
    ego_xy: np.ndarray  # (E, 2)
    ego_yaw: np.ndarray  # (E,)
    # Annotations, one row per (key frame, object)
    ann_sample: np.ndarray  # (A,) int index into sample_t
    ann_inst: np.ndarray  # (A,) scene-local instance id
    ann_group: np.ndarray  # (A,) int index into GROUPS
    ann_attr: np.ndarray  # (A,) int index into ATTRIBUTES
    ann_xy: np.ndarray  # (A, 2) world
    ann_yaw: np.ndarray  # (A,)
    ann_size: np.ndarray  # (A, 3) w, l, h
    ann_vis: np.ndarray  # (A,) 1..4 visibility bin
    can: dict[str, np.ndarray] = field(default_factory=dict)  # optional CAN bus signals

    @property
    def duration(self) -> float:
        return float(self.sample_t[-1]) if len(self.sample_t) else 0.0


def _load(table_dir: Path, name: str) -> list[dict]:
    with open(table_dir / f"{name}.json") as f:
        return json.load(f)


def _load_can(can_dir: Path, scene_name: str, start_utime: int) -> dict[str, np.ndarray]:
    """Vehicle monitor (2 Hz) and IMU-derived accel (subsampled) from the CAN bus expansion."""
    out: dict[str, np.ndarray] = {}
    vm_path = can_dir / f"{scene_name}_vehicle_monitor.json"
    if vm_path.exists():
        vm = json.loads(vm_path.read_text())
        if vm:
            out["vm_t"] = np.array([(m["utime"] - start_utime) / 1e6 for m in vm])
            for k in (
                "vehicle_speed",
                "brake",
                "brake_switch",
                "left_signal",
                "right_signal",
                "steering",
                "gear_position",
                "throttle",
                "yaw_rate",
            ):
                out[k] = np.array([m[k] for m in vm], dtype=float)
    pose_path = can_dir / f"{scene_name}_pose.json"
    if pose_path.exists():
        pose = json.loads(pose_path.read_text())[::5]  # 50 Hz -> 10 Hz
        if pose:
            out["pose_t"] = np.array([(m["utime"] - start_utime) / 1e6 for m in pose])
            out["pose_vel"] = np.array([m["vel"][0] for m in pose], dtype=float)
            out["pose_accel"] = np.array([m["accel"][0] for m in pose], dtype=float)
    return out


def build_logs(data_dir: Path, version: str = "v1.0-trainval") -> dict[str, ScenarioLog]:
    tdir = data_dir / version
    log.info("loading tables from %s", tdir)
    logs = {r["token"]: r for r in _load(tdir, "log")}
    scenes = _load(tdir, "scene")
    samples = _load(tdir, "sample")
    sensors = {r["token"]: r["channel"] for r in _load(tdir, "sensor")}
    calib = {r["token"]: sensors[r["sensor_token"]] for r in _load(tdir, "calibrated_sensor")}
    cats = {r["token"]: CATEGORY_GROUPS[r["name"]] for r in _load(tdir, "category")}
    attr_names = {r["token"]: r["name"] for r in _load(tdir, "attribute")}
    inst_cat = {r["token"]: cats[r["category_token"]] for r in _load(tdir, "instance")}

    sample_scene = {s["token"]: s["scene_token"] for s in samples}
    by_scene_samples: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_scene_samples[s["scene_token"]].append(s)

    log.info("indexing sample_data / ego_pose")
    ego_pose = {r["token"]: r for r in _load(tdir, "ego_pose")}
    lidar_poses: dict[str, list[tuple[int, list, list]]] = defaultdict(list)
    cam_front: dict[str, str] = {}
    for sd in _load(tdir, "sample_data"):
        ch = calib[sd["calibrated_sensor_token"]]
        if ch == "LIDAR_TOP":
            p = ego_pose[sd["ego_pose_token"]]
            lidar_poses[sample_scene[sd["sample_token"]]].append((p["timestamp"], p["translation"], p["rotation"]))
        elif ch == "CAM_FRONT" and sd["is_key_frame"]:
            cam_front[sd["sample_token"]] = sd["filename"]
    del ego_pose

    log.info("grouping annotations")
    anns_by_sample: dict[str, list[dict]] = defaultdict(list)
    for a in _load(tdir, "sample_annotation"):
        anns_by_sample[a["sample_token"]].append(a)

    can_dir = data_dir / "can_bus"
    out: dict[str, ScenarioLog] = {}
    for sc in scenes:
        ss = sorted(by_scene_samples[sc["token"]], key=lambda s: s["timestamp"])
        t0 = ss[0]["timestamp"]
        poses = sorted(lidar_poses[sc["token"]])
        inst_ids: dict[str, int] = {}
        rows = []
        for si, s in enumerate(ss):
            for a in anns_by_sample.get(s["token"], []):
                iid = inst_ids.setdefault(a["instance_token"], len(inst_ids))
                attr = attr_names[a["attribute_tokens"][0]] if a["attribute_tokens"] else "none"
                rows.append(
                    (
                        si,
                        iid,
                        GROUPS.index(inst_cat[a["instance_token"]]),
                        ATTRIBUTES.index(attr),
                        a["translation"][0],
                        a["translation"][1],
                        quat_to_yaw(a["rotation"]),
                        a["size"][0],
                        a["size"][1],
                        a["size"][2],
                        int(a["visibility_token"]),
                    )
                )
        arr = np.array(rows, dtype=float) if rows else np.zeros((0, 11))
        lg = logs[sc["log_token"]]
        out[sc["name"]] = ScenarioLog(
            name=sc["name"],
            token=sc["token"],
            location=lg["location"],
            log_name=lg["logfile"],
            date=lg["date_captured"],
            start_utime=t0,
            description=sc["description"],
            sample_t=np.array([(s["timestamp"] - t0) / 1e6 for s in ss]),
            cam_front=[cam_front.get(s["token"], "") for s in ss],
            ego_t=np.array([(p[0] - t0) / 1e6 for p in poses]),
            ego_xy=np.array([p[1][:2] for p in poses]),
            ego_yaw=np.array([quat_to_yaw(p[2]) for p in poses]),
            ann_sample=arr[:, 0].astype(np.int16),
            ann_inst=arr[:, 1].astype(np.int32),
            ann_group=arr[:, 2].astype(np.int8),
            ann_attr=arr[:, 3].astype(np.int8),
            ann_xy=arr[:, 4:6].astype(np.float32),
            ann_yaw=arr[:, 6].astype(np.float32),
            ann_size=arr[:, 7:10].astype(np.float32),
            ann_vis=arr[:, 10].astype(np.int8),
            can=_load_can(can_dir, sc["name"], t0) if can_dir.exists() else {},
        )
    log.info("built %d scenario logs", len(out))
    return out


def save_logs(logs: dict[str, ScenarioLog], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_logs(path: Path) -> dict[str, ScenarioLog]:
    with open(path, "rb") as f:
        return pickle.load(f)
