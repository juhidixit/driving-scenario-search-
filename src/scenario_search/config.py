"""Filesystem locations. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SCENARIO_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = Path(os.environ.get("NUSCENES_ROOT", ROOT / "data" / "nuscenes"))
VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-trainval")
ARTIFACTS_DIR = Path(os.environ.get("SCENARIO_ARTIFACTS", ROOT / "artifacts"))
MODELS_DIR = Path(os.environ.get("SCENARIO_MODELS", ROOT / "models"))

CLIP_DIR = Path(os.environ.get("CLIP_MODEL_DIR", MODELS_DIR / "clip-vit-base-patch16"))
TEXT_DIR = Path(os.environ.get("TEXT_MODEL_DIR", MODELS_DIR / "all-MiniLM-L6-v2"))

LOGS_FILE = "scenario_logs.pkl"
TAGS_FILE = "scenario_tags.json"
CLIP_FILE = "clip_frames.npz"
CAPTION_EMB_FILE = "caption_emb.npz"
