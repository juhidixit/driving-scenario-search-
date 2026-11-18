"""The human scene descriptions are evaluation labels: indexing / search code must not read them."""

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "scenario_search"
ALLOWED = {"nuscenes_io.py", "build_measures.py"}  # copy the field through for display only


def test_description_not_used_by_index_or_search():
    offenders = []
    for f in PKG.glob("*.py"):
        if f.name in ALLOWED:
            continue
        if re.search(r"""["']description["']|\.description\b""", f.read_text()):
            offenders.append(f.name)
    assert offenders == []
