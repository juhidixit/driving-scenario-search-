"""FastAPI service: search, scenario detail, similar scenarios, OpenSCENARIO export, demo UI."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from . import __version__, config
from .search import MODES, SearchEngine, Weights

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Driving Scenario Search", version=__version__)


@lru_cache(maxsize=1)
def engine() -> SearchEngine:
    w = Weights()
    wf = config.ROOT / "configs" / "fusion_weights.json"
    if wf.exists():
        w = Weights(**json.loads(wf.read_text()))
    use_models = os.environ.get("SCENARIO_USE_LEARNED", "0") == "1"
    return SearchEngine(
        config.ARTIFACTS_DIR,
        use_clip=os.environ.get("SCENARIO_NO_CLIP", "0") != "1",
        weights=w,
        tag_models=config.ARTIFACTS_DIR / "tag_models.pkl" if use_models else None,
    )


class Health(BaseModel):
    status: str
    scenarios: int
    clip_frames: int
    version: str


@app.get("/health", response_model=Health)
def health() -> Health:
    e = engine()
    return Health(
        status="ok",
        scenarios=len(e.names),
        clip_frames=int(len(e.frame_emb)) if e.clip is not None else 0,
        version=__version__,
    )


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=300),
    k: int = Query(12, ge=1, le=100),
    mode: str = "hybrid",
    location: str | None = None,
) -> dict:
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    return engine().search(q, k=k, mode=mode, location=location)


@app.get("/api/scenario/{name}")
def scenario(name: str) -> dict:
    e = engine()
    if name not in e.idx:
        raise HTTPException(404, "unknown scenario")
    return e.scenario(name)


@app.get("/api/similar/{name}")
def similar(name: str, k: int = Query(8, ge=1, le=50), use: str = "both") -> list[dict]:
    e = engine()
    if name not in e.idx:
        raise HTTPException(404, "unknown scenario")
    return e.similar(name, k=k, use=use)


@app.get("/api/export/{name}.xosc")
def export_xosc(name: str) -> Response:
    from .export_xosc import to_xosc

    logs = _logs()
    if name not in logs:
        raise HTTPException(404, "unknown scenario")
    return Response(
        to_xosc(logs[name]),
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{name}.xosc"'},
    )


@lru_cache(maxsize=1)
def _logs():
    from .nuscenes_io import load_logs

    return load_logs(config.ARTIFACTS_DIR / config.LOGS_FILE)


@app.get("/frames/{path:path}")
def frame(path: str) -> FileResponse:
    root = config.DATA_DIR.resolve()
    p = (root / path).resolve()
    if not str(p).startswith(str(root)) or not p.suffix.lower() == ".jpg" or not p.exists():
        raise HTTPException(404, "frame not available")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()
