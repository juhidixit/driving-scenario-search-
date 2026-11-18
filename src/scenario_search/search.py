"""Hybrid scenario search: structured tags + CLIP frames + caption embeddings.

Scoring for a query ``q`` parsed into terms ``t_1..t_k`` (see :mod:`scenario_search.query`):

* each term gets a per-scenario score in [0, 1]:
  - tags with a rule-based detector use the detector score (or a learned tag model, if loaded),
  - appearance-only tags (rain, bridge, ...) use the CLIP score of the tag's own phrase,
  - numeric filters use a soft step function,
  - negated terms use ``1 - score``;
* the structured score is the mean term score;
* the open-vocabulary score is a min-max normalised mix of CLIP (whole query vs. frames, max
  over frames) and caption similarity (MiniLM);
* ``hybrid = w * structured + (1 - w) * open_vocab``; with no parsed terms the
  open-vocabulary score is used alone.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .ontology import TAG_BY_NAME, TAGS
from .query import ParsedQuery, numeric_score, parse

CLIP_TEMPLATES = (
    "a dashcam photo of {}.",
    "a photo from a car's front camera showing {}.",
    "{}",
)
MODES = ("hybrid", "structured", "clip", "caption", "open_vocab")


@dataclass
class Weights:
    structured: float = 0.7  # share of the structured score in hybrid mode
    clip: float = 0.6  # share of CLIP inside the open-vocabulary score


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


class SearchEngine:
    def __init__(
        self,
        artifacts: Path = config.ARTIFACTS_DIR,
        use_clip: bool = True,
        use_text: bool = True,
        weights: Weights | None = None,
        tag_models: Path | None = None,
    ):
        self.artifacts = Path(artifacts)
        rec = json.loads((self.artifacts / "scenario_measures.json").read_text())
        tags = json.loads((self.artifacts / config.TAGS_FILE).read_text())
        self.names: list[str] = sorted(rec)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.records = rec
        self.captions = [tags[n]["caption"] for n in self.names]
        self.rule_tags = [t.name for t in TAGS if t.has_rule]
        self.rule = np.array([[tags[n]["scores"][t] for t in self.rule_tags] for n in self.names])
        self.weights = weights or Weights()
        self.learned = None
        if tag_models and Path(tag_models).exists():
            with open(tag_models, "rb") as f:
                self.learned = pickle.load(f)

        self.clip = None
        if use_clip and (self.artifacts / config.CLIP_FILE).exists():
            z = np.load(self.artifacts / config.CLIP_FILE)
            self.frame_scene = np.array([self.idx[s] for s in z["scene"]])
            self.frame_t = z["t"]
            self.frame_file = z["path"]
            self.frame_emb = z["emb"].astype(np.float32)
            counts = np.bincount(self.frame_scene, minlength=len(self.names))
            self.scene_clip = np.zeros((len(self.names), self.frame_emb.shape[1]), np.float32)
            np.add.at(self.scene_clip, self.frame_scene, self.frame_emb)
            self.scene_clip /= np.clip(counts[:, None], 1, None)
            self.has_frames = counts > 0
            from .embed import clip_encoder

            self.clip = clip_encoder()
        self.text = None
        if use_text and (self.artifacts / config.CAPTION_EMB_FILE).exists():
            z = np.load(self.artifacts / config.CAPTION_EMB_FILE)
            order = [list(z["names"]).index(n) for n in self.names]
            self.caption_emb = z["emb"][order]
            from .embed import text_encoder

            self.text = text_encoder()
        self._clip_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- component scores
    def clip_text(self, text: str) -> np.ndarray:
        if text not in self._clip_cache:
            e = self.clip.encode_text([t.format(text) for t in CLIP_TEMPLATES]).mean(0)
            self._clip_cache[text] = e / np.linalg.norm(e)
        return self._clip_cache[text]

    def clip_scores(self, text: str, return_frames: bool = False):
        """Max cosine similarity over each scenario's frames."""
        sims = self.frame_emb @ self.clip_text(text)
        best = np.full(len(self.names), -1.0, dtype=np.float32)
        np.maximum.at(best, self.frame_scene, sims)
        best[~self.has_frames] = float(np.median(best[self.has_frames]))
        if not return_frames:
            return best
        arg = np.full(len(self.names), -1)
        order = np.argsort(sims)  # ascending, so the last write per scene is its best frame
        arg[self.frame_scene[order]] = order
        return best, arg

    def caption_scores(self, text: str) -> np.ndarray:
        return self.caption_emb @ self.text.encode([text])[0]

    def tag_scores(self, tag: str) -> np.ndarray:
        """Per-scenario score in [0, 1] for one tag."""
        if self.learned is not None and tag in self.learned["models"]:
            return self.learned_tag_scores(tag)
        if tag in self.rule_tags:
            return self.rule[:, self.rule_tags.index(tag)]
        if self.clip is not None:  # appearance tag: CLIP zero-shot on the tag phrase
            return _minmax(self.clip_scores(TAG_BY_NAME[tag].phrases[0]))
        return np.full(len(self.names), 0.5)

    def learned_tag_scores(self, tag: str) -> np.ndarray:
        from .learned import predict

        return predict(self.learned, tag, self)

    def structured_scores(self, pq: ParsedQuery) -> tuple[np.ndarray | None, dict]:
        parts, detail = [], {}
        for term in pq.terms:
            s = self.tag_scores(term.tag)
            s = 1.0 - s if term.negated else s
            parts.append(s)
            detail[("NOT " if term.negated else "") + term.tag] = s
        for f in pq.numeric:
            vals = np.array([self.records[n]["measures"].get(f.measure, 0.0) for n in self.names])
            s = np.array([numeric_score(v, f.op, f.value) for v in vals])
            parts.append(s)
            detail[f"{f.measure} {f.op} {f.value:g}"] = s
        if not parts:
            return None, detail
        return np.mean(parts, axis=0), detail

    def open_vocab_scores(self, text: str) -> np.ndarray | None:
        comps, ws = [], []
        if self.clip is not None:
            comps.append(_minmax(self.clip_scores(text)))
            ws.append(self.weights.clip)
        if self.text is not None:
            comps.append(_minmax(self.caption_scores(text)))
            ws.append(1 - self.weights.clip)
        if not comps:
            return None
        return np.average(comps, axis=0, weights=ws)

    def score(self, query: str, mode: str = "hybrid") -> tuple[np.ndarray, ParsedQuery, dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        pq = parse(query)
        n = len(self.names)
        if mode == "clip":
            return _minmax(self.clip_scores(query)), pq, {}
        if mode == "caption":
            return _minmax(self.caption_scores(query)), pq, {}
        if mode == "open_vocab":
            return self.open_vocab_scores(query), pq, {}
        structured, detail = self.structured_scores(pq)
        if mode == "structured":
            return (structured if structured is not None else np.zeros(n)), pq, detail
        ov = self.open_vocab_scores(query)
        if structured is None:
            return (ov if ov is not None else np.zeros(n)), pq, detail
        if ov is None:
            return structured, pq, detail
        w = self.weights.structured
        return w * structured + (1 - w) * ov, pq, detail

    # -------------------------------------------------------------------- public API
    def search(self, query: str, k: int = 10, mode: str = "hybrid", location: str | None = None) -> dict:
        t0 = time.perf_counter()
        scores, pq, detail = self.score(query, mode)
        mask = np.ones(len(self.names), bool)
        if location:
            mask = np.array([self.records[n]["location"].startswith(location) for n in self.names])
        order = [i for i in np.argsort(-scores) if mask[i]][:k]
        best_frames = None
        if self.clip is not None:
            _, best_frames = self.clip_scores(query, return_frames=True)
        results = [self._result(i, float(scores[i]), detail, pq, best_frames) for i in order]
        return {
            "query": query,
            "mode": mode,
            "parsed": pq.explain(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "results": results,
        }

    def _result(self, i: int, score: float, detail: dict, pq: ParsedQuery, best_frames) -> dict:
        name = self.names[i]
        rec = self.records[name]
        spans = {t.tag: rec["spans"][t.tag] for t in pq.terms if t.tag in rec["spans"] and not t.negated}
        frame = None
        if best_frames is not None and best_frames[i] >= 0:
            j = best_frames[i]
            frame = {"t": float(self.frame_t[j]), "file": str(self.frame_file[j])}
        if spans and rec["frames"]:
            t_mid = float(np.mean(next(iter(spans.values()))))
            t, f = min(rec["frames"], key=lambda fr: abs(fr[0] - t_mid))
            frame = {"t": t, "file": f}
        return {
            "scene": name,
            "score": round(score, 4),
            "location": rec["location"],
            "caption": self.captions[i],
            "term_scores": {k: round(float(v[i]), 3) for k, v in detail.items()},
            "spans": spans,
            "frame": frame,
        }

    def scenario(self, name: str) -> dict:
        rec = self.records[name]
        i = self.idx[name]
        return {
            "scene": name,
            "location": rec["location"],
            "date": rec["date"],
            "caption": self.captions[i],
            "tags": {
                t: round(float(self.rule[i, j]), 3) for j, t in enumerate(self.rule_tags) if self.rule[i, j] >= 0.5
            },
            "measures": {k: round(v, 2) for k, v in rec["measures"].items()},
            "spans": rec["spans"],
            "track": rec["track"],
            "frames": rec["frames"],
        }

    # ------------------------------------------------------------- query by example
    def _similarity_matrix(self) -> np.ndarray:
        if not hasattr(self, "_sim_feats"):
            keys = [t.measure for t in TAGS if t.has_rule]
            keys = sorted(set(keys))
            x = np.array([[self.records[n]["measures"].get(k, 0.0) for k in keys] for n in self.names])
            x = np.sign(x) * np.log1p(np.abs(x))
            x = (x - x.mean(0)) / (x.std(0) + 1e-6)
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
            feats = [x]
            if self.clip is not None:
                c = self.scene_clip - self.scene_clip.mean(0)
                feats.append(c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9))
            self._sim_feats = np.concatenate(feats, 1) / np.sqrt(len(feats))
        return self._sim_feats

    def similar(self, name: str, k: int = 10, use: str = "both") -> list[dict]:
        f = self._similarity_matrix()
        n_struct = f.shape[1] if self.clip is None else f.shape[1] - self.scene_clip.shape[1]
        if use == "structured":
            f = f[:, :n_struct]
        elif use == "clip" and self.clip is not None:
            f = f[:, n_struct:]
        i = self.idx[name]
        sims = f @ f[i]
        sims[i] = -np.inf
        order = np.argsort(-sims)[:k]
        return [
            {
                "scene": self.names[j],
                "score": round(float(sims[j]), 4),
                "location": self.records[self.names[j]]["location"],
                "caption": self.captions[j],
            }
            for j in order
        ]
