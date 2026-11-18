"""Compute measures for every scenario log and write them to ``scenario_measures.json``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .features import compute_measures
from .hdmap import load_map
from .nuscenes_io import load_logs

log = logging.getLogger(__name__)


def build(artifacts: Path = config.ARTIFACTS_DIR, data_dir: Path = config.DATA_DIR) -> dict:
    logs = load_logs(artifacts / config.LOGS_FILE)
    map_dir = str(data_dir / "maps" / "expansion")
    out = {}
    for name, slog in logs.items():
        r = compute_measures(slog, load_map(map_dir, slog.location))
        frames = [i for i, f in enumerate(slog.cam_front) if f]
        out[name] = {
            "location": slog.location,
            "log": slog.log_name,
            "description": slog.description,  # display + eval only; never indexed
            "date": slog.date,
            "duration_s": round(slog.duration, 2),
            "frames": [[round(float(slog.sample_t[i]), 2), slog.cam_front[i]] for i in frames],
            **r,
        }
    path = artifacts / "scenario_measures.json"
    path.write_text(json.dumps(out))
    log.info("wrote %s (%d scenes)", path, len(out))
    return out
