"""Point-in-layer queries against the nuScenes map expansion (v1.3) with shapely STRtrees."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

POLY_LAYERS = ("road_segment", "lane", "lane_connector", "ped_crossing", "walkway", "carpark_area", "stop_line")


class HDMap:
    """Vectorised membership queries for a single nuScenes map location."""

    def __init__(self, path: Path):
        raw = json.loads(Path(path).read_text())
        nodes = {n["token"]: (n["x"], n["y"]) for n in raw["node"]}
        polys = {}
        for p in raw["polygon"]:
            pts = [nodes[t] for t in p["exterior_node_tokens"]]
            if len(pts) >= 3:
                poly = Polygon(pts)
                polys[p["token"]] = poly if poly.is_valid else poly.buffer(0)
        self.layers: dict[str, dict] = {}
        for layer in POLY_LAYERS:
            recs = [r for r in raw.get(layer, []) if r.get("polygon_token") in polys]
            geoms = [polys[r["polygon_token"]] for r in recs]
            self.layers[layer] = {
                "records": recs,
                "geoms": geoms,
                "tree": shapely.STRtree(geoms) if geoms else None,
            }
        self.connectivity: dict[str, list[str]] = {
            k: v.get("outgoing", []) for k, v in raw.get("connectivity", {}).items()
        }
        self.lane_tokens = {r["token"] for r in self.layers["lane"]["records"]}

    # ------------------------------------------------------------------ queries
    def contains(self, layer: str, xy: np.ndarray) -> list[list[int]]:
        """For each point return the indices of ``layer`` records that contain it."""
        info = self.layers[layer]
        out: list[list[int]] = [[] for _ in range(len(xy))]
        if info["tree"] is None or len(xy) == 0:
            return out
        pts = shapely.points(np.asarray(xy, dtype=float))
        pi, gi = info["tree"].query(pts, predicate="intersects")
        for p, g in zip(pi, gi):
            out[int(p)].append(int(g))
        return out

    def any_contains(self, layer: str, xy: np.ndarray) -> np.ndarray:
        return np.array([len(c) > 0 for c in self.contains(layer, xy)], dtype=bool)

    def near(self, layer: str, xy: np.ndarray, radius: float) -> np.ndarray:
        """True where a point is within ``radius`` metres of any polygon of ``layer``."""
        info = self.layers[layer]
        out = np.zeros(len(xy), dtype=bool)
        if info["tree"] is None or len(xy) == 0:
            return out
        pi, _ = info["tree"].query(shapely.points(np.asarray(xy, dtype=float)), predicate="dwithin", distance=radius)
        out[np.unique(pi)] = True
        return out

    def in_intersection(self, xy: np.ndarray) -> np.ndarray:
        recs = self.layers["road_segment"]["records"]
        return np.array(
            [any(recs[i]["is_intersection"] for i in c) for c in self.contains("road_segment", xy)],
            dtype=bool,
        )

    def on_road(self, xy: np.ndarray) -> np.ndarray:
        return self.any_contains("lane", xy) | self.any_contains("lane_connector", xy) | self.in_intersection(xy)

    def lane_of(self, xy: np.ndarray) -> list[str | None]:
        """Token of the (non-connector) lane containing each point, or None."""
        recs = self.layers["lane"]["records"]
        return [recs[c[0]]["token"] if c else None for c in self.contains("lane", xy)]

    def stop_lines_near(self, xy: np.ndarray, radius: float = 4.0) -> list[set[str]]:
        info = self.layers["stop_line"]
        out: list[set[str]] = [set() for _ in range(len(xy))]
        if info["tree"] is None or len(xy) == 0:
            return out
        pts = shapely.points(np.asarray(xy, dtype=float))
        pi, gi = info["tree"].query(pts, predicate="dwithin", distance=radius)
        for p, g in zip(pi, gi):
            out[int(p)].add(info["records"][int(g)]["stop_line_type"])
        return out

    def lanes_reachable(self, lane: str, hops: int = 3) -> set[str]:
        """Lanes reachable by following lane / connector successors (and predecessors)."""
        seen, frontier = {lane}, {lane}
        preds: dict[str, list[str]] = _predecessors(id(self), self.connectivity)
        for _ in range(hops):
            nxt = set()
            for t in frontier:
                nxt.update(self.connectivity.get(t, []))
                nxt.update(preds.get(t, []))
            frontier = nxt - seen
            seen |= nxt
        return seen


_PRED_CACHE: dict[int, dict[str, list[str]]] = {}


def _predecessors(key: int, conn: dict[str, list[str]]) -> dict[str, list[str]]:
    if key not in _PRED_CACHE:
        preds: dict[str, list[str]] = defaultdict(list)
        for k, outs in conn.items():
            for o in outs:
                preds[o].append(k)
        _PRED_CACHE[key] = dict(preds)
    return _PRED_CACHE[key]


@lru_cache(maxsize=8)
def load_map(map_dir: str, location: str) -> HDMap:
    return HDMap(Path(map_dir) / f"{location}.json")
