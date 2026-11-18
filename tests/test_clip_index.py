"""build_clip_index + SearchEngine CLIP path with a fake encoder (no torch / model download needed)."""

import json

import numpy as np
import pytest

from scenario_search import config, embed, index
from scenario_search.ontology import score_tags
from scenario_search.search import SearchEngine

PIL = pytest.importorskip("PIL.Image")


class FakeClip:
    """'Images' are embedded by mean colour; text 'red' / 'blue' maps onto the same axes."""

    def encode_images(self, images):
        e = np.array([np.asarray(im, float).mean((0, 1)) for im in images]) + 1e-3
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def encode_text(self, texts):
        e = np.array([[1.0, 0, 0] if "red" in t else [0, 0, 1.0] for t in texts])
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_clip_index_and_search(tmp_path, monkeypatch):
    data, art = tmp_path / "data", tmp_path / "art"
    (data / "samples").mkdir(parents=True)
    art.mkdir()
    rec, tags = {}, {}
    for name, colour in (("scene-red", (220, 10, 10)), ("scene-blue", (10, 10, 220))):
        frames = []
        for k in range(8):
            rel = f"samples/{name}_{k}.jpg"
            PIL.new("RGB", (16, 16), colour).save(data / rel)
            frames.append([k * 0.5, rel])
        rec[name] = {
            "location": "boston-seaport",
            "log": name,
            "date": "d",
            "frames": frames,
            "measures": {},
            "spans": {},
            "track": {},
        }
        tags[name] = {"scores": score_tags({}), "caption": name}
    (art / "scenario_measures.json").write_text(json.dumps(rec))
    (art / config.TAGS_FILE).write_text(json.dumps(tags))
    monkeypatch.setattr(embed, "clip_encoder", lambda: FakeClip())

    index.build_clip_index(art, data, every_n=4)
    z = np.load(art / config.CLIP_FILE)
    assert len(z["emb"]) == 4 and set(z["scene"]) == {"scene-red", "scene-blue"}
    index.build_clip_index(art, data, every_n=2)  # incremental: reuses the 4 cached frames
    assert len(np.load(art / config.CLIP_FILE)["emb"]) == 8

    eng = SearchEngine(art, use_clip=True, use_text=False)
    assert eng.search("something red", k=1, mode="clip")["results"][0]["scene"] == "scene-red"
    assert eng.search("something blue", k=1)["results"][0]["scene"] == "scene-blue"
