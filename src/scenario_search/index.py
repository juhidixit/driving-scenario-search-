"""Build the search index artifacts from scenario measures and CAM_FRONT key frames.

Artifacts (all under ``SCENARIO_ARTIFACTS``):

* ``scenario_measures.json``  measures, evidence spans, ego track, frame list (build_measures)
* ``scenario_tags.json``      rule tag scores + caption per scenario
* ``caption_emb.npz``         MiniLM embeddings of the captions
* ``clip_frames.npz``         CLIP image embeddings of sampled CAM_FRONT key frames
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from . import config
from .captions import caption
from .ontology import load_thresholds, score_tags

log = logging.getLogger(__name__)
THRESHOLDS_FILE = config.ROOT / "configs" / "thresholds.json"


def build_tags(artifacts: Path = config.ARTIFACTS_DIR, thresholds_path: Path = THRESHOLDS_FILE) -> dict:
    measures = json.loads((artifacts / "scenario_measures.json").read_text())
    thr = load_thresholds(thresholds_path)
    out = {}
    for name, rec in measures.items():
        scores = score_tags(rec["measures"], thr)
        out[name] = {"scores": scores, "caption": caption(scores, rec["measures"])}
    (artifacts / config.TAGS_FILE).write_text(json.dumps(out))
    log.info("tagged %d scenarios", len(out))
    return out


def build_caption_embeddings(artifacts: Path = config.ARTIFACTS_DIR) -> None:
    from .embed import text_encoder

    tags = json.loads((artifacts / config.TAGS_FILE).read_text())
    names = sorted(tags)
    emb = text_encoder().encode([tags[n]["caption"] for n in names])
    np.savez_compressed(artifacts / config.CAPTION_EMB_FILE, names=np.array(names), emb=emb)
    log.info("caption embeddings %s", emb.shape)


def build_clip_index(
    artifacts: Path = config.ARTIFACTS_DIR, data_dir: Path = config.DATA_DIR, every_n: int = 4, batch: int = 32
) -> None:
    """Embed every ``every_n``-th CAM_FRONT key frame (2 s apart at the default of 4)."""
    from PIL import Image

    from .embed import clip_encoder

    measures = json.loads((artifacts / "scenario_measures.json").read_text())
    jobs = []
    for name in sorted(measures):
        frames = measures[name]["frames"]
        for t, rel in frames[::every_n]:
            if (data_dir / rel).exists():
                jobs.append((name, t, rel))
    # Reuse embeddings from a previous (partial) run so the index can be grown incrementally.
    cache: dict[str, np.ndarray] = {}
    if (artifacts / config.CLIP_FILE).exists():
        z = np.load(artifacts / config.CLIP_FILE)
        cache = {str(f): e for f, e in zip(z["path"], z["emb"])}
    todo = [j for j in jobs if j[2] not in cache]
    log.info("embedding %d new frames (%d cached)", len(todo), len(jobs) - len(todo))
    enc = clip_encoder()
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        imgs = [Image.open(data_dir / rel).convert("RGB") for _, _, rel in chunk]
        for (_, _, rel), e in zip(chunk, enc.encode_images(imgs)):
            cache[rel] = e
        if (i // batch) % 20 == 0:
            log.info("%d/%d frames (%.1f img/s)", i + len(imgs), len(todo), (i + len(imgs)) / (time.time() - t0))
    embs = [cache[rel][None] for _, _, rel in jobs]
    np.savez_compressed(
        artifacts / config.CLIP_FILE,
        scene=np.array([j[0] for j in jobs]),
        t=np.array([j[1] for j in jobs], dtype=np.float32),
        path=np.array([j[2] for j in jobs]),
        emb=np.concatenate(embs).astype(np.float16),
    )
