"""End-to-end evaluation of scenario search against held-out human scene descriptions.

Protocol (all decisions are made on the dev split; numbers are reported on test):

1. Split the 850 nuScenes trainval scenes ~50/50 by *capture day*, see ``split_of``.
2. Tune per-tag rule thresholds on dev (max F1 vs. description labels) -> configs/thresholds.json.
3. Train weakly supervised tag classifiers on dev labels (optional "learned" methods).
4. Tune the hybrid fusion weights on dev (grid search, mean AP).
5. Evaluate every method on test: AP, nDCG@10, P@10 per query, averaged over paraphrases
   and queries. Queries with fewer than MIN_POS test positives are skipped.
6. Query-by-example: description-tag Jaccard of top-5 neighbours vs. random.

Run:  python -m eval.run_eval  (writes results/RESULTS.md and results/results.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.ground_truth import GT_PATTERNS, day_key, gt_tags, split_of  # noqa: E402
from scenario_search import config  # noqa: E402
from scenario_search.index import THRESHOLDS_FILE, build_caption_embeddings, build_tags  # noqa: E402
from scenario_search.ontology import TAG_BY_NAME  # noqa: E402
from scenario_search.search import SearchEngine, Weights  # noqa: E402

log = logging.getLogger("eval")
MIN_POS = 5
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- metrics
def average_precision(scores: np.ndarray, rel: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    r = rel[order]
    if r.sum() == 0:
        return float("nan")
    hits = np.cumsum(r)
    return float(np.sum(hits[r] / (np.arange(len(r))[r] + 1)) / r.sum())


def ndcg_at(scores: np.ndarray, rel: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    gains = rel[order] / np.log2(np.arange(2, len(order) + 2))
    ideal = np.sort(rel)[::-1][:k] / np.log2(np.arange(2, min(k, len(rel)) + 2))
    return float(gains.sum() / ideal.sum()) if ideal.sum() else float("nan")


def precision_at(scores: np.ndarray, rel: np.ndarray, k: int = 10) -> float:
    return float(rel[np.argsort(-scores, kind="stable")[:k]].mean())


def metrics(scores, rel) -> dict:
    return {"AP": average_precision(scores, rel), "nDCG@10": ndcg_at(scores, rel), "P@10": precision_at(scores, rel)}


# --------------------------------------------------------------------------- helpers
def labels_for(engine: SearchEngine) -> tuple[dict, np.ndarray, np.ndarray]:
    desc = [engine.records[n]["description"] for n in engine.names]
    tags = [gt_tags(d) for d in desc]
    labels = {t: np.array([t in s for s in tags]) for t in GT_PATTERNS}
    days = [day_key(engine.records[n]["location"], engine.records[n]["date"]) for n in engine.names]
    split = np.array([split_of(d) for d in days])
    groups = np.array(days)
    return labels, split, groups


def query_set(labels: dict, mask: np.ndarray, queries: dict) -> list[dict]:
    out = []
    for group in ("single", "compositional"):
        for q in queries[group]:
            rel = np.all([labels[t] for t in q["gt"]], axis=0)
            if rel[mask].sum() < MIN_POS:
                continue
            out.append({"group": group, "gt": q["gt"], "queries": q["queries"], "rel": rel})
    return out


def evaluate(
    engine: SearchEngine, qs: list[dict], mask: np.ndarray, mode: str, score_cache: dict | None = None
) -> list[dict]:
    rows = []
    for q in qs:
        per = []
        for text in q["queries"]:
            key = (text, mode)
            if score_cache is not None and key in score_cache:
                s = score_cache[key]
            else:
                s = engine.score(text, mode)[0]
                if score_cache is not None:
                    score_cache[key] = s
            per.append(metrics(s[mask], q["rel"][mask]))
        rows.append(
            {
                "group": q["group"],
                "gt": "+".join(q["gt"]),
                "n_pos": int(q["rel"][mask].sum()),
                **{m: float(np.mean([p[m] for p in per])) for m in per[0]},
            }
        )
    return rows


def random_rows(qs: list[dict], mask: np.ndarray, seeds: int = 200) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for q in qs:
        rel = q["rel"][mask]
        ms = [metrics(rng.random(len(rel)), rel) for _ in range(seeds)]
        rows.append(
            {
                "group": q["group"],
                "gt": "+".join(q["gt"]),
                "n_pos": int(rel.sum()),
                **{m: float(np.mean([x[m] for x in ms])) for m in ms[0]},
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for group in ("single", "compositional", "all"):
        sel = [r for r in rows if group == "all" or r["group"] == group]
        out[group] = {m: round(float(np.mean([r[m] for r in sel])), 4) for m in ("AP", "nDCG@10", "P@10")}
        out[group]["n_queries"] = len(sel)
    return out


# --------------------------------------------------------------------------- tuning
def tune_thresholds(engine: SearchEngine, labels: dict, dev: np.ndarray) -> dict:
    thresholds = {}
    for tag, y in labels.items():
        t = TAG_BY_NAME.get(tag)
        if t is None or not t.has_rule or y[dev].sum() < MIN_POS:
            continue
        v = np.array([engine.records[n]["measures"].get(t.measure, 0.0) for n in engine.names])[dev]
        yy = y[dev]
        sign = 1 if t.scale > 0 else -1
        cands = np.unique(np.quantile(v, np.linspace(0.02, 0.98, 49)))
        best = (-1.0, t.threshold)
        for c in cands:
            pred = (v * sign) > (c * sign)
            tp = (pred & yy).sum()
            f1 = 2 * tp / max(1, pred.sum() + yy.sum())
            if f1 > best[0] + 1e-9:
                best = (f1, float(c))
        # Midpoint between the chosen value and the next one keeps integer counts stable.
        thresholds[tag] = round(best[1] + (0.5 * sign if np.all(np.mod(v, 1) == 0) else 0.0), 4)
    return thresholds


def tune_weights(engine: SearchEngine, qs: list[dict], dev: np.ndarray) -> Weights:
    best = (-1.0, Weights())
    cache: dict = {}
    for wc in (0.3, 0.5, 0.7, 0.9):
        engine.weights = Weights(structured=1.0, clip=wc)
        ov = summarize(evaluate(engine, qs, dev, "open_vocab"))["all"]["AP"]
        for ws in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
            engine.weights = Weights(structured=ws, clip=wc)
            ap = summarize(evaluate(engine, qs, dev, "hybrid"))["all"]["AP"]
            log.info("dev weights structured=%.1f clip=%.1f -> hybrid mAP %.4f (open-vocab %.4f)", ws, wc, ap, ov)
            if ap > best[0]:
                best = (ap, Weights(structured=ws, clip=wc))
    del cache
    return best[1]


# --------------------------------------------------------------------------- QbE
def qbe_eval(engine: SearchEngine, labels: dict, mask: np.ndarray, k: int = 5) -> dict:
    tags = [frozenset(t for t, y in labels.items() if y[i]) for i in range(len(engine.names))]
    idx = np.where(mask)[0]
    names = set(engine.names[i] for i in idx)

    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0

    out = {}
    rng = np.random.default_rng(0)
    for use in ("structured", "clip", "both"):
        if use != "structured" and engine.clip is None:
            continue
        js = []
        for i in idx:
            nb = [r["scene"] for r in engine.similar(engine.names[i], k=60, use=use) if r["scene"] in names][:k]
            js.append(np.mean([jac(tags[i], tags[engine.idx[n]]) for n in nb]))
        out[use] = round(float(np.mean(js)), 4)
    rnd = [np.mean([jac(tags[i], tags[j]) for j in rng.choice(idx, k, replace=False)]) for i in idx]
    out["random"] = round(float(np.mean(rnd)), 4)
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    queries = json.loads((ROOT / "eval" / "queries.json").read_text())

    # 1-2. thresholds on dev, then rebuild tags / captions with them
    if not (args.artifacts / config.TAGS_FILE).exists():
        build_tags(args.artifacts, None)
    engine = SearchEngine(args.artifacts, use_clip=False, use_text=False)
    labels, split, groups = labels_for(engine)
    dev, test = split == "dev", split == "test"
    log.info(
        "dev %d scenes / %d days, test %d scenes / %d days",
        dev.sum(),
        len(set(groups[dev])),
        test.sum(),
        len(set(groups[test])),
    )
    thr = tune_thresholds(engine, labels, dev)
    THRESHOLDS_FILE.parent.mkdir(exist_ok=True)
    THRESHOLDS_FILE.write_text(json.dumps(thr, indent=1, sort_keys=True))
    build_tags(args.artifacts, THRESHOLDS_FILE)
    build_caption_embeddings(args.artifacts)

    engine = SearchEngine(args.artifacts, use_clip=not args.no_clip, use_text=True)
    dev_qs, test_qs = query_set(labels, dev, queries), query_set(labels, test, queries)

    # 3. learned tag models (dev labels only)
    from scenario_search.learned import train

    learned = train(engine, labels, dev, groups, use_clip=engine.clip is not None)
    with open(args.artifacts / "tag_models.pkl", "wb") as f:
        pickle.dump(learned, f)

    # 4. fusion weights on dev
    weights = tune_weights(engine, dev_qs, dev)
    engine.weights = weights
    log.info("chosen weights %s", weights)
    (ROOT / "configs" / "fusion_weights.json").write_text(
        json.dumps({"structured": weights.structured, "clip": weights.clip}, indent=1)
    )

    # 5. test evaluation
    methods: dict[str, list[dict]] = {"random": random_rows(test_qs, test)}
    modes = ["caption", "structured", "hybrid"] + (["clip", "open_vocab"] if engine.clip is not None else [])
    for m in modes:
        methods[f"zero-shot {m}"] = evaluate(engine, test_qs, test, m)
    engine.learned = learned
    for m in ("structured", "hybrid"):
        methods[f"learned {m}"] = evaluate(engine, test_qs, test, m)
    engine.learned = None

    # 6. query by example + latency
    qbe = qbe_eval(engine, labels, test)
    lat = []
    for q in test_qs:
        for text in q["queries"]:
            t0 = time.perf_counter()
            engine.search(text, k=10)
            lat.append((time.perf_counter() - t0) * 1000)

    summary = {m: summarize(r) for m, r in methods.items()}
    result = {
        "protocol": {
            "dev_scenes": int(dev.sum()),
            "test_scenes": int(test.sum()),
            "dev_days": len(set(groups[dev])),
            "test_days": len(set(groups[test])),
            "min_test_positives": MIN_POS,
            "weights": weights.__dict__,
            "n_clip_frames": int(len(engine.frame_emb)) if engine.clip is not None else 0,
        },
        "summary": summary,
        "per_query": methods,
        "query_by_example_jaccard@5": qbe,
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
            "n": len(lat),
        },
        "learned_cv": learned["cv"],
        "thresholds": thr,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(result, indent=1))
    (RESULTS / "RESULTS.md").write_text(render_md(result))
    print((RESULTS / "RESULTS.md").read_text())


def render_md(r: dict) -> str:
    p = r["protocol"]
    lines = [
        "# Results",
        "",
        "Generated by `python -m eval.run_eval`. Do not edit by hand.",
        "",
        f"Test split: **{p['test_scenes']} scenes from {p['test_days']} capture days** (dev: {p['dev_scenes']} scenes, "
        f"{p['dev_days']} days; split by day so weather and lighting of a day never leak across). "
        "Ground truth = regex tags over the human-written nuScenes scene descriptions, which the index never reads. "
        f"Queries with fewer than {p['min_test_positives']} test positives are skipped. "
        f"Fusion weights tuned on dev: structured={p['weights']['structured']}, clip={p['weights']['clip']}. "
        f"CLIP frames indexed: {p['n_clip_frames']}.",
        "",
        "## Retrieval quality (test split, mean over queries)",
        "",
        "| Method | mAP (all) | nDCG@10 (all) | P@10 (all) | mAP single-concept | mAP compositional |",
        "|---|---|---|---|---|---|",
    ]
    for m, s in r["summary"].items():
        lines.append(
            f"| {m} | {s['all']['AP']:.3f} | {s['all']['nDCG@10']:.3f} | {s['all']['P@10']:.3f} | "
            f"{s['single']['AP']:.3f} | {s['compositional']['AP']:.3f} |"
        )
    n = r["summary"]["random"]
    lines += [
        "",
        f"Queries: {n['single']['n_queries']} single-concept, {n['compositional']['n_queries']} compositional, "
        "2 paraphrases each for single-concept queries.",
        "",
        "## Per-query AP (test)",
        "",
    ]
    methods = list(r["per_query"])
    lines.append("| Query (GT tags) | test positives | " + " | ".join(methods) + " |")
    lines.append("|---|---|" + "---|" * len(methods))
    rows = r["per_query"]["random"]
    for i, row in enumerate(rows):
        vals = [r["per_query"][m][i]["AP"] for m in methods]
        best = max(vals)
        cells = [f"**{v:.2f}**" if v == best else f"{v:.2f}" for v in vals]
        lines.append(f"| {row['gt']} | {row['n_pos']} | " + " | ".join(cells) + " |")
    q = r["query_by_example_jaccard@5"]
    lines += [
        "",
        "## Query by example (test)",
        "",
        "Mean Jaccard overlap between the description tags of a scene and those of its top-5 most similar "
        "test scenes (higher = neighbours look alike to a human).",
        "",
        "| Similarity features | Jaccard@5 |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in q.items()]
    lat = r["latency_ms"]
    lines += [
        "",
        "## Latency",
        "",
        f"End-to-end `search()` latency over {lat['n']} test queries on a 4-vCPU CPU box (includes CLIP + "
        f"MiniLM text encoding): **p50 {lat['p50']} ms, p95 {lat['p95']} ms**.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
