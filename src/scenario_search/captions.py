"""Deterministic natural-language captions generated from tag scores and measures.

Captions make the structured index searchable with a sentence-embedding model and are
shown in the UI as a readable summary of each scenario.
"""

from __future__ import annotations

from .ontology import TAGS

COUNT_MEASURE = {
    "pedestrians": "n_peds",
    "many_pedestrians": "n_peds",
    "bus": "n_bus",
    "truck": "n_truck",
    "cones": "n_cone",
    "barriers": "front_barrier",
    "parked_cars": "front_parked_vehicles",
    "cyclist": "n_cyclists",
    "motorcycle": "front_motorcycle",
    "ped_crossing": "n_peds_crossing",
}
REDUNDANT = {
    "many_pedestrians": "pedestrians",
    "turn_across_traffic": None,
    "cross_intersection": "intersection",
}


def caption(scores: dict[str, float], measures: dict, threshold: float = 0.5) -> str:
    on = {t for t, s in scores.items() if s >= threshold}
    for specific, generic in REDUNDANT.items():
        if specific in on and generic in on:
            on.discard(generic)
    light = "Night-time" if "night" in on else "Daytime" if "daytime" in on else "Dusk"
    city = "Singapore" if measures.get("is_singapore") else "Boston"
    speed = measures.get("max_speed_kmh", 0.0)
    parts = [f"{light} drive in {city}, top speed {speed:.0f} km/h."]
    man = [t.label for t in TAGS if t.name in on and t.kind == "maneuver"]
    mp = [t.label for t in TAGS if t.name in on and t.kind == "map"]
    if man or mp:
        parts.append("The ego vehicle " + ", ".join(man + mp) + ".")
    ag = []
    for t in TAGS:
        if t.name in on and t.kind == "agent":
            n = measures.get(COUNT_MEASURE.get(t.name, ""), 0)
            ag.append(f"{t.label} ({int(n)})" if n else t.label)
    if ag:
        parts.append("Scene: " + "; ".join(ag) + ".")
    return " ".join(parts)
