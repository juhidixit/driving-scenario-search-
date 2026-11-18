"""Rule-based natural-language query parser.

Turns "unprotected left turn at night with no pedestrians, faster than 30 km/h" into

* tag terms:       turn_across_traffic, night, NOT pedestrians
* numeric filters: max_speed_kmh > 30
* residual text:   whatever is left, which goes to the CLIP and caption encoders

The parser is deterministic and needs no model or network access. It runs in
~0.1 ms per query, so it never dominates latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ontology import TAGS

NEGATIONS = ("without", "no", "not", "never", "zero", "free of", "isn't", "aren't", "excluding")
_NEG_RE = re.compile(r"(?:^|\b)(" + "|".join(re.escape(n) for n in NEGATIONS) + r")\s+(?:any\s+|a\s+|an\s+)?$")

NUMERIC_NOUNS = {
    "pedestrians": "n_peds",
    "pedestrian": "n_peds",
    "peds": "n_peds",
    "people": "n_peds",
    "cars": "n_vehicles",
    "vehicles": "n_vehicles",
    "buses": "n_bus",
    "bus": "n_bus",
    "trucks": "n_truck",
    "cyclists": "n_cyclists",
    "cones": "n_cone",
    "parked cars": "front_parked_vehicles",
    "km/h": "max_speed_kmh",
    "kmh": "max_speed_kmh",
    "kph": "max_speed_kmh",
    "mph": "max_speed_mph",
    "seconds stopped": "longest_stop_s",
}
_OPS = {
    "at least": ">=",
    "more than": ">",
    "over": ">",
    "above": ">",
    "faster than": ">",
    "greater than": ">",
    ">=": ">=",
    ">": ">",
    "at most": "<=",
    "fewer than": "<",
    "less than": "<",
    "under": "<",
    "below": "<",
    "slower than": "<",
    "<=": "<=",
    "<": "<",
}
_NUM_RE = re.compile(
    r"("
    + "|".join(sorted(map(re.escape, _OPS), key=len, reverse=True))
    + r")\s*(\d+(?:\.\d+)?)\s*("
    + "|".join(sorted(map(re.escape, NUMERIC_NOUNS), key=len, reverse=True))
    + r")\b"
)
STOPWORDS = {
    "a",
    "an",
    "the",
    "with",
    "and",
    "of",
    "in",
    "at",
    "on",
    "to",
    "where",
    "while",
    "ego",
    "vehicle",
    "car",
    "scenario",
    "scenarios",
    "scene",
    "scenes",
    "show",
    "me",
    "find",
    "for",
    "some",
    "any",
    "is",
    "are",
    "that",
    "which",
    "there",
    "then",
    "from",
    "near",
    "by",
    "it",
    "we",
    "our",
    "us",
    "drive",
    "driving",
    "clips",
    "clip",
    "or",
}


@dataclass
class Term:
    tag: str
    negated: bool = False
    span: tuple[int, int] = (0, 0)


@dataclass
class NumericFilter:
    measure: str
    op: str
    value: float


@dataclass
class ParsedQuery:
    raw: str
    terms: list[Term] = field(default_factory=list)
    numeric: list[NumericFilter] = field(default_factory=list)
    residual: str = ""

    @property
    def is_structured(self) -> bool:
        return bool(self.terms or self.numeric)

    def explain(self) -> dict:
        return {
            "tags": [("NOT " if t.negated else "") + t.tag for t in self.terms],
            "filters": [f"{f.measure} {f.op} {f.value:g}" for f in self.numeric],
            "residual": self.residual,
        }


_PHRASES = sorted(((p, t.name) for t in TAGS for p in t.phrases), key=lambda x: len(x[0]), reverse=True)


def normalize(q: str) -> str:
    q = q.lower().replace("’", "'")
    q = re.sub(r"[^\w\s/<>=.'-]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def parse(query: str) -> ParsedQuery:
    q = normalize(query)
    out = ParsedQuery(raw=query)
    taken = [False] * len(q)

    for m in _NUM_RE.finditer(q):
        op, val, noun = m.group(1), float(m.group(2)), m.group(3)
        measure = NUMERIC_NOUNS[noun]
        if measure == "max_speed_mph":
            measure, val = "max_speed_kmh", val * 1.609
        out.numeric.append(NumericFilter(measure, _OPS[op], val))
        for i in range(m.start(), m.end()):
            taken[i] = True

    for phrase, tag in _PHRASES:
        for m in re.finditer(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", q):
            if any(taken[m.start() : m.end()]):
                continue
            prefix = q[max(0, m.start() - 20) : m.start()]
            neg_m = _NEG_RE.search(prefix)
            negated = bool(neg_m)
            start = m.start() - (len(prefix) - neg_m.start(1)) if neg_m else m.start()
            for i in range(start, m.end()):
                taken[i] = True
            if not any(t.tag == tag for t in out.terms):
                out.terms.append(Term(tag, negated, (m.start(), m.end())))

    residual = "".join(c if not taken[i] else " " for i, c in enumerate(q))
    words = [w for w in residual.split() if w not in STOPWORDS and w not in NEGATIONS and len(w) > 1]
    out.residual = " ".join(words)
    out.terms.sort(key=lambda t: t.span)
    return out


def numeric_score(value: float, op: str, target: float) -> float:
    """Soft 0/1 satisfaction of a numeric constraint (5% tolerance band)."""
    import math

    width = max(0.5, abs(target) * 0.05)
    z = (value - target) / width
    if op in ("<", "<="):
        z = -z
    if op in (">=", "<="):
        z += 0.5
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z * 2))))
