"""Weakly supervised tag classifiers.

Labels come from regexes over the human scene descriptions (``eval/ground_truth.py``),
**on the dev split only**. Features are the scenario measures (log-scaled) and, when
available, the mean CLIP embedding of the scenario's front-camera frames. One
L2-regularised logistic regression per tag, with C chosen by grouped (per-log)
cross-validation on dev.
"""

from __future__ import annotations

import numpy as np


def measure_keys(records: dict) -> list[str]:
    first = next(iter(records.values()))["measures"]
    return sorted(k for k, v in first.items() if isinstance(v, int | float))


def features(engine, keys: list[str], use_clip: bool) -> np.ndarray:
    x = np.array([[engine.records[n]["measures"].get(k, 0.0) for k in keys] for n in engine.names])
    x = np.sign(x) * np.log1p(np.abs(x))
    if use_clip and getattr(engine, "clip", None) is not None:
        x = np.concatenate([x, engine.scene_clip * 10.0], axis=1)
    return x


def train(
    engine,
    labels: dict[str, np.ndarray],
    train_mask: np.ndarray,
    groups: np.ndarray,
    use_clip: bool = True,
    min_pos: int = 8,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keys = measure_keys(engine.records)
    x = features(engine, keys, use_clip)[train_mask]
    g = groups[train_mask]
    models, cv = {}, {}
    for tag, y_all in labels.items():
        y = y_all[train_mask].astype(int)
        if y.sum() < min_pos:
            continue
        best = (-1.0, 0.1)
        for c in (0.003, 0.01, 0.03, 0.1, 0.3):
            aps = []
            for tr, va in GroupKFold(n_splits=4).split(x, y, g):
                if y[va].sum() == 0 or y[tr].sum() == 0:
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
                m.fit(x[tr], y[tr])
                aps.append(average_precision_score(y[va], m.predict_proba(x[va])[:, 1]))
            if aps and np.mean(aps) > best[0]:
                best = (float(np.mean(aps)), c)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=2000, class_weight="balanced"))
        m.fit(x, y)
        models[tag] = m
        cv[tag] = {"cv_ap": round(best[0], 3), "C": best[1], "n_pos": int(y.sum())}
    return {"models": models, "keys": keys, "use_clip": use_clip, "cv": cv}


def predict(learned: dict, tag: str, engine) -> np.ndarray:
    cache = engine.__dict__.setdefault("_learned_x", {})
    key = learned["use_clip"]
    if key not in cache:
        cache[key] = features(engine, learned["keys"], learned["use_clip"])
    return learned["models"][tag].predict_proba(cache[key])[:, 1]
