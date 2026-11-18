"""Weak ground truth from the human-written nuScenes scene descriptions.

Every nuScenes scene carries a short free-text description written by the annotation
team ("Night, rain, turn left, jaywalker"). We turn those into binary relevance labels
with conservative regexes. The search index never sees these descriptions, so they are a
held-out test of whether sensor-derived search finds what a person saw.

Caveat (documented in the README): descriptions are incomplete. A missing tag does not
mean the thing is absent, so absolute precision is underestimated. Comparisons between
methods on the same labels remain fair.
"""

from __future__ import annotations

import hashlib
import re

GT_PATTERNS: dict[str, str] = {
    "night": r"\bnight\b|\bin dark\b|\bdark\b",
    "rain": r"\brain|\bwet ground\b|umbrella",
    "parking_lot": r"parking lot|parking bay|car ?park",
    "wait_at_intersection": r"\bwait(ing)?\b|stop(ped)? at (red light|intersection|stop line)|red light",
    "turn_left": r"turn(ing)? left|left turn",
    "turn_right": r"turn(ing)? right|right turn",
    "cross_intersection": r"cross(ing)? (a |the )?(busy |highway-like )?intersection",
    "intersection": r"intersection",
    "bus": r"\bbus(es)?\b(?! stops?\b)|bendy bus|school bus",
    "truck": r"\btrucks?\b|heavy veh",
    "construction_zone": r"construction|excavator|crane|digger|\bworkers?\b",
    "cones": r"\bcones?\b",
    "barriers": r"barrier|barricade",
    "parked_cars": r"parked (cars?|vehicles?|vans?)|parked car",
    "pedestrians": r"\bpeds?\b|pedestrian|jaywalker|\bpeople\b",
    "many_pedestrians": r"(many|lots of|crowd of) (peds|pedestrians|people)|crowd",
    "ped_crossing": r"(peds?|pedestrians?) crossing|crossing (crosswalk|road|street)",
    "jaywalker": r"jaywalk",
    "cyclist": r"cyclist|bicyclist|bike on sidewalk|follow cyclist|bicycle(s)? (riding|on)",
    "parked_bicycles": r"parked (bicycles?|bikes?)",
    "bicycle_rack": r"(bicycle|bike) racks?",
    "motorcycle": r"motorcycl|motorcyl|motorbike",
    "motorcyclist": r"scooter|motorcyclist|motorcycle driving|moped",
    "personal_mobility": r"personal mobility|e-scooter|segway",
    "bus_stop": r"bus stops?",
    "lane_change": r"lane change|chang(e|ing) lanes?|switch lanes?",
    "overtaken": r"overtaking us|passed by|passing by|overtakes us",
    "dense_traffic": r"dense traffic|congestion|heavy traffic|many cars|busy (intersection|road|street)",
    "high_speed": r"high speed|highway",
    "bridge": r"bridge|flyover",
    "nature": r"\bnature\b|\btrees\b|bushes|greenery|park\b(?! ?ing)",
    "construction_vehicle": r"construction vehicle|excavator|crane|digger|concrete truck",
    "trailer": r"trailer",
    "animal": r"animal|\bdog\b|\bbird\b|\bpet\b",
}


def gt_tags(description: str) -> set[str]:
    d = description.lower()
    return {tag for tag, pat in GT_PATTERNS.items() if re.search(pat, d)}


def day_key(location: str, date: str) -> str:
    return f"{location}/{date}"


SPLIT_SALT = "3"  # picked once, before running any method, so the halves are ~50/50 in scenes


def split_of(day: str) -> str:
    """Deterministic dev/test split by *capture day* (location + date).

    Splitting by day rather than by scene or drive means weather and lighting of a given day
    never leak from dev into test (e.g. a rainy afternoon is entirely in one half).
    """
    h = int(hashlib.sha1(f"{SPLIT_SALT}:{day}".encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "test"
