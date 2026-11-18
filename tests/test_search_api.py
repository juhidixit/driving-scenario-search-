import json

import pytest
from fastapi.testclient import TestClient

from scenario_search.features import compute_measures
from scenario_search.hdmap import HDMap
from scenario_search.index import build_tags
from scenario_search.nuscenes_io import save_logs
from scenario_search.search import SearchEngine


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, logs, map_dir):
    d = tmp_path_factory.mktemp("artifacts")
    hd = HDMap(map_dir / "boston-seaport.json")
    rec = {}
    for name, slog in logs.items():
        rec[name] = {
            "location": slog.location,
            "log": slog.log_name,
            "date": slog.date,
            "description": "",
            "duration_s": slog.duration,
            "frames": [],
            **compute_measures(slog, hd),
        }
    (d / "scenario_measures.json").write_text(json.dumps(rec))
    save_logs(logs, d / "scenario_logs.pkl")
    build_tags(d, None)
    return d


def test_structured_search_ranks_expected_scene_first(artifacts):
    eng = SearchEngine(artifacts, use_clip=False, use_text=False)
    assert eng.search("left turn", k=1)["results"][0]["scene"] == "scene-left"
    assert eng.search("waiting at a red light", k=1)["results"][0]["scene"] == "scene-stop"
    assert eng.search("crowd of pedestrians", k=1)["results"][0]["scene"] == "scene-crowd"
    top = eng.search("left turn without pedestrians", k=1)["results"][0]
    assert top["scene"] == "scene-left" and "NOT pedestrians" in top["term_scores"]


def test_similar_excludes_self(artifacts):
    eng = SearchEngine(artifacts, use_clip=False, use_text=False)
    sims = eng.similar("scene-straight", k=3)
    assert "scene-straight" not in [s["scene"] for s in sims]
    assert sims[0]["scene"] == "scene-crowd"  # same ego path


def test_api(artifacts, monkeypatch):
    from scenario_search import api, config

    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setenv("SCENARIO_NO_CLIP", "1")
    api.engine.cache_clear()
    api._logs.cache_clear()
    monkeypatch.setattr(api, "engine", lambda: SearchEngine(artifacts, use_clip=False, use_text=False))
    c = TestClient(api.app)
    assert c.get("/health").json()["scenarios"] == 4
    r = c.get("/api/search", params={"q": "left turn", "k": 2}).json()
    assert r["results"][0]["scene"] == "scene-left" and r["parsed"]["tags"] == ["turn_left"]
    assert c.get("/api/scenario/scene-left").json()["measures"]["turn_left_deg"] > 80
    assert c.get("/api/scenario/nope").status_code == 404
    assert c.get("/api/search", params={"q": "x", "mode": "bad"}).status_code == 422
    assert c.get("/api/export/scene-stop.xosc").text.startswith("<?xml")
    assert c.get("/frames/../../etc/passwd").status_code == 404
    assert "Driving Scenario Search" in c.get("/").text
