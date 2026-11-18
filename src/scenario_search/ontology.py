"""Scenario tag ontology: what each tag means, how it is scored, and the words that name it.

Each tag maps one measure (see :mod:`scenario_search.features`) to a soft score in
[0, 1] with a logistic ramp around a threshold. Scores stay graded so ranking still
works when several tags are combined in a query. Thresholds are tuned on the dev split
only (``eval/tune_thresholds.py``) and stored in ``configs/thresholds.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tag:
    name: str
    kind: str  # maneuver | map | agent | condition | appearance (no rule; image/learned only)
    label: str  # caption phrase
    measure: str
    threshold: float
    scale: float  # logistic width; negative => "less than threshold" tags
    phrases: tuple[str, ...]

    @property
    def has_rule(self) -> bool:
        return bool(self.measure)

    def score(self, measures: dict, threshold: float | None = None) -> float:
        if not self.measure:
            raise ValueError(f"tag {self.name!r} has no rule-based detector")
        thr = self.threshold if threshold is None else threshold
        v = float(measures.get(self.measure, 0.0))
        z = (v - thr) / self.scale
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


T = Tag
TAGS: tuple[Tag, ...] = (
    # ---------------------------------------------------------------- ego maneuvers
    T(
        "turn_left",
        "maneuver",
        "turns left",
        "turn_left_deg",
        45,
        8,
        ("turn left", "turns left", "turning left", "left turn", "left-hand turn", "hang a left"),
    ),
    T(
        "turn_right",
        "maneuver",
        "turns right",
        "turn_right_deg",
        45,
        8,
        ("turn right", "turns right", "turning right", "right turn", "right-hand turn", "hang a right"),
    ),
    T(
        "turn_across_traffic",
        "maneuver",
        "turns across the oncoming lanes",
        "turn_across_traffic_deg",
        45,
        8,
        (
            "unprotected turn",
            "unprotected left",
            "turn across traffic",
            "turn across oncoming traffic",
            "turning across traffic",
        ),
    ),
    T(
        "oncoming_during_turn",
        "agent",
        "with oncoming vehicles during the turn",
        "oncoming_during_turn",
        0.5,
        0.25,
        ("oncoming traffic", "oncoming vehicle", "oncoming vehicles", "oncoming car", "oncoming cars"),
    ),
    T(
        "u_turn",
        "maneuver",
        "makes a U-turn",
        "max_abs_heading_deg",
        150,
        10,
        ("u-turn", "u turn", "uturn", "turn around", "turns around"),
    ),
    T(
        "lane_change",
        "maneuver",
        "changes lanes",
        "lane_change_evidence",
        0.5,
        0.15,
        (
            "lane change",
            "change lane",
            "changes lane",
            "changing lane",
            "switch lane",
            "switches lane",
            "merge into",
            "merging",
        ),
    ),
    T(
        "stop",
        "maneuver",
        "comes to a stop",
        "n_stops",
        0.5,
        0.15,
        ("comes to a stop", "stops", "stopping", "come to a stop", "brakes to a stop", "halt"),
    ),
    T(
        "start_from_stop",
        "maneuver",
        "moves off from a stop",
        "n_starts",
        0.5,
        0.15,
        (
            "start from stop",
            "starts moving",
            "pulls away",
            "pull away",
            "moves off",
            "start at intersection",
            "starts at intersection",
            "light turns green",
        ),
    ),
    T(
        "stationary",
        "maneuver",
        "is stationary",
        "frac_stopped",
        0.8,
        0.05,
        ("stationary", "standing still", "not moving", "ego stopped", "stationary ego vehicle"),
    ),
    T(
        "wait_at_intersection",
        "maneuver",
        "waits at an intersection",
        "longest_wait_near_intersection_s",
        2.0,
        0.6,
        (
            "wait at intersection",
            "waits at intersection",
            "waiting at intersection",
            "wait at red light",
            "waiting at red light",
            "stopped at red light",
            "red light",
            "stopped at intersection",
            "wait at stop line",
            "waiting at a light",
            "stop at intersection",
        ),
    ),
    T(
        "cross_intersection",
        "maneuver",
        "drives through an intersection",
        "crosses_intersection",
        0.5,
        0.15,
        (
            "cross intersection",
            "crosses intersection",
            "crossing intersection",
            "through intersection",
            "through the intersection",
            "go through intersection",
            "drive through intersection",
        ),
    ),
    T(
        "high_speed",
        "maneuver",
        "drives at high speed",
        "max_speed_kmh",
        45,
        4,
        ("high speed", "fast", "speeding", "highway", "expressway", "at speed"),
    ),
    T(
        "hard_brake",
        "maneuver",
        "brakes hard",
        "min_accel_mps2",
        -2.5,
        -0.4,
        ("hard brake", "hard braking", "brakes hard", "emergency brake", "sudden stop", "slams on the brakes"),
    ),
    T("reverse", "maneuver", "reverses", "frac_reverse", 0.1, 0.03, ("reverse", "reversing", "backing up", "backs up")),
    T(
        "following",
        "maneuver",
        "follows a lead vehicle",
        "frac_following",
        0.5,
        0.1,
        ("following", "follow a car", "follow bus", "lead vehicle", "car ahead", "behind a car", "car following"),
    ),
    T(
        "cut_in",
        "agent",
        "with a vehicle cutting in",
        "n_cut_ins",
        0.5,
        0.15,
        ("cut in", "cuts in", "cutting in", "cut-in", "merges in front"),
    ),
    T(
        "overtaken",
        "agent",
        "is overtaken by another vehicle",
        "n_overtaken_by",
        0.5,
        0.15,
        ("overtaking us", "overtakes us", "overtaken", "passed by", "passing us", "car passing by"),
    ),
    # ---------------------------------------------------------------- map context
    T(
        "intersection",
        "map",
        "at an intersection",
        "enters_intersection",
        0.5,
        0.15,
        ("intersection", "junction", "crossroads"),
    ),
    T(
        "crosswalk",
        "map",
        "over a crosswalk",
        "crosswalks_crossed",
        0.5,
        0.15,
        ("crosswalk", "zebra crossing", "pedestrian crossing area"),
    ),
    T(
        "parking_lot",
        "map",
        "next to a parking lot",
        "frac_near_carpark",
        0.15,
        0.04,
        ("parking lot", "parking lots", "car park", "carpark", "parking garage", "parking area"),
    ),
    T("stop_sign", "map", "near a stop sign", "near_stop_sign", 0.5, 0.15, ("stop sign",)),
    T(
        "traffic_light",
        "map",
        "near a traffic light",
        "near_traffic_light",
        0.5,
        0.15,
        ("traffic light", "traffic lights", "signalized", "signalised"),
    ),
    # ---------------------------------------------------------------- other agents
    T(
        "pedestrians",
        "agent",
        "with pedestrians",
        "n_peds",
        2.5,
        0.8,
        ("pedestrian", "pedestrians", "peds", "ped", "people", "person", "walkers"),
    ),
    T(
        "many_pedestrians",
        "agent",
        "with many pedestrians",
        "n_peds",
        20,
        4,
        (
            "many pedestrians",
            "many peds",
            "lots of pedestrians",
            "lots of peds",
            "crowd",
            "crowded",
            "busy sidewalk",
            "lots of people",
            "many people",
        ),
    ),
    T(
        "ped_crossing",
        "agent",
        "with pedestrians crossing in front",
        "n_peds_crossing",
        0.5,
        0.15,
        (
            "pedestrian crossing",
            "pedestrians crossing",
            "peds crossing",
            "ped crossing",
            "crossing the road",
            "crossing the street",
            "people crossing",
            "crossing pedestrians",
            "peds cross",
            "pedestrians cross",
            "people cross",
            "cross the street",
            "cross the road",
        ),
    ),
    T(
        "jaywalker",
        "agent",
        "with a jaywalker",
        "n_jaywalkers",
        0.5,
        0.15,
        ("jaywalker", "jaywalkers", "jaywalking", "crossing mid-block"),
    ),
    T(
        "cyclist",
        "agent",
        "with a cyclist",
        "n_cyclists",
        0.5,
        0.15,
        ("cyclist", "cyclists", "bicyclist", "biker", "person on a bike", "riding a bicycle", "bike rider"),
    ),
    T(
        "parked_bicycles",
        "agent",
        "past parked bicycles",
        "n_parked_bicycles",
        0.5,
        0.15,
        ("parked bicycle", "parked bicycles", "parked bike", "parked bikes"),
    ),
    T(
        "bicycle_rack",
        "agent",
        "past a bicycle rack",
        "front_bicycle_rack",
        0.5,
        0.15,
        ("bicycle rack", "bike rack", "bicycle racks", "bike racks"),
    ),
    T(
        "motorcycle",
        "agent",
        "with motorcycles",
        "front_motorcycle",
        0.5,
        0.15,
        ("motorcycle", "motorcycles", "motorbike", "motorbikes"),
    ),
    T(
        "motorcyclist",
        "agent",
        "with a motorcycle or scooter rider",
        "n_motorcyclists",
        0.5,
        0.15,
        ("motorcyclist", "motorcyclists", "scooter", "scooters", "moped", "motorbike rider"),
    ),
    T(
        "personal_mobility",
        "agent",
        "with an e-scooter or personal mobility device",
        "n_scooter",
        0.5,
        0.15,
        ("e-scooter", "personal mobility", "kick scooter", "segway", "skateboard"),
    ),
    T("bus", "agent", "with a bus", "n_bus", 0.5, 0.15, ("bus", "buses", "bendy bus", "coach")),
    T("truck", "agent", "with a truck", "n_truck", 0.5, 0.15, ("truck", "trucks", "lorry", "heavy vehicle", "semi")),
    T("trailer", "agent", "with a trailer", "n_trailer", 0.5, 0.15, ("trailer", "trailers")),
    T(
        "construction_zone",
        "agent",
        "through a construction zone",
        "construction_score_raw",
        12,
        3,
        ("construction", "construction zone", "construction site", "road works", "roadworks", "work zone"),
    ),
    T(
        "construction_vehicle",
        "agent",
        "with construction vehicles",
        "n_construction_vehicle",
        0.5,
        0.15,
        ("construction vehicle", "excavator", "crane", "digger", "bulldozer"),
    ),
    T(
        "construction_worker",
        "agent",
        "with construction workers",
        "n_construction_worker",
        0.5,
        0.15,
        ("construction worker", "construction workers", "workers", "road workers"),
    ),
    T("cones", "agent", "with traffic cones", "n_cone", 2.5, 0.8, ("cone", "cones", "traffic cones")),
    T("barriers", "agent", "with barriers", "front_barrier", 2.5, 0.8, ("barrier", "barriers", "barricade")),
    T(
        "parked_cars",
        "agent",
        "past parked cars",
        "front_parked_vehicles",
        2.5,
        0.8,
        ("parked car", "parked cars", "parked vehicles", "parked vehicle", "cars parked"),
    ),
    T(
        "dense_traffic",
        "agent",
        "in dense traffic",
        "mean_traffic_vehicles",
        6,
        1.5,
        ("dense traffic", "heavy traffic", "congestion", "traffic jam", "busy traffic", "rush hour"),
    ),
    T(
        "light_traffic",
        "agent",
        "in light traffic",
        "mean_traffic_vehicles",
        1.0,
        -0.4,
        ("low traffic", "light traffic", "empty road", "no traffic", "quiet road"),
    ),
    T(
        "emergency_vehicle",
        "agent",
        "with an emergency vehicle",
        "n_emergency_vehicle",
        0.5,
        0.15,
        ("ambulance", "police car", "emergency vehicle", "fire truck"),
    ),
    T(
        "police_officer",
        "agent",
        "with a police officer",
        "n_police_officer",
        0.5,
        0.15,
        ("police officer", "traffic police", "cop", "officer directing traffic"),
    ),
    T("animal", "agent", "with an animal", "n_animal", 0.5, 0.15, ("animal", "dog", "cat", "bird", "pet")),
    T("child", "agent", "with a child", "n_child", 0.5, 0.15, ("child", "children", "kid", "kids")),
    T("stroller", "agent", "with a stroller", "n_stroller", 0.5, 0.15, ("stroller", "pram", "baby carriage")),
    T("wheelchair", "agent", "with a wheelchair user", "n_wheelchair", 0.5, 0.15, ("wheelchair",)),
    # ---------------------------------------------------------------- conditions
    T(
        "night",
        "condition",
        "at night",
        "sun_elevation_deg",
        -6,
        -1.5,
        ("night", "nighttime", "night time", "at night", "dark", "after dark", "in the dark"),
    ),
    T(
        "daytime",
        "condition",
        "in daylight",
        "sun_elevation_deg",
        6,
        1.5,
        ("day", "daytime", "daylight", "during the day", "sunny"),
    ),
    # ------------------------------------------------ appearance (camera-only) tags
    T(
        "rain",
        "appearance",
        "in the rain",
        "",
        0,
        1,
        ("rain", "raining", "rainy", "wet road", "wet roads", "umbrella", "umbrellas", "downpour"),
    ),
    T("bridge", "appearance", "on a bridge", "", 0, 1, ("bridge", "overpass", "flyover", "viaduct")),
    T(
        "nature",
        "appearance",
        "past trees and greenery",
        "",
        0,
        1,
        ("nature", "trees", "greenery", "park", "vegetation", "forest"),
    ),
    T("bus_stop", "appearance", "past a bus stop", "", 0, 1, ("bus stop", "bus stops", "bus shelter")),
    T("boston", "condition", "in Boston", "is_boston", 0.5, 0.15, ("boston", "seaport", "us city")),
    T(
        "singapore",
        "condition",
        "in Singapore",
        "is_singapore",
        0.5,
        0.15,
        ("singapore", "left-hand traffic", "one-north", "holland village", "queenstown"),
    ),
)

TAG_BY_NAME = {t.name: t for t in TAGS}


def load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def score_tags(measures: dict, thresholds: dict[str, float] | None = None) -> dict[str, float]:
    thresholds = thresholds or {}
    return {t.name: round(t.score(measures, thresholds.get(t.name)), 4) for t in TAGS if t.has_rule}
