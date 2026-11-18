import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scenario_search.export_xosc import to_xosc


def test_export_structure(logs):
    xml = to_xosc(logs["scene-stop"])
    root = ET.fromstring(xml)
    assert root.tag == "OpenSCENARIO"
    names = [o.get("name") for o in root.iter("ScenarioObject")]
    assert names[0] == "Ego" and any(n.startswith("pedestrian_") for n in names)
    verts = list(next(root.iter("Polyline")).iter("Vertex"))
    times = [float(v.get("time")) for v in verts]
    assert times == sorted(times) and len(times) > 10


def test_export_validates_against_official_schema(logs):
    lxml = pytest.importorskip("lxml.etree")
    sg = pytest.importorskip("scenariogeneration")
    xsd = Path(sg.__file__).parents[1] / "schemas" / "OpenSCENARIO_1_2.xsd"
    if not xsd.exists():
        pytest.skip("schema not bundled with this scenariogeneration version")
    schema = lxml.XMLSchema(lxml.parse(str(xsd)))
    doc = lxml.fromstring(to_xosc(logs["scene-left"]).encode())
    assert schema.validate(doc), schema.error_log
