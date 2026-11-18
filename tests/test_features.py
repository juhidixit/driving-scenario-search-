from scenario_search.features import compute_measures
from scenario_search.hdmap import HDMap


def measures(logs, map_dir, name):
    return compute_measures(logs[name], HDMap(map_dir / "boston-seaport.json"))


def test_left_turn_detected(logs, map_dir):
    r = measures(logs, map_dir, "scene-left")
    m = r["measures"]
    assert 80 < m["turn_left_deg"] < 100
    assert m["turn_right_deg"] < 5
    assert m["turn_across_traffic_deg"] == m["turn_left_deg"]  # right-hand traffic in Boston
    assert m["enters_intersection"] == 1.0
    assert "turn_left" in r["spans"]


def test_straight_drive_has_no_turn_or_stop(logs, map_dir):
    m = measures(logs, map_dir, "scene-straight")["measures"]
    assert m["turn_left_deg"] < 5 and m["turn_right_deg"] < 5
    assert m["n_stops"] == 0
    assert 35 < m["max_speed_kmh"] < 37  # 10 m/s
    assert m["frac_near_carpark"] > 0.3  # passes the car park polygon


def test_wait_at_traffic_light(logs, map_dir):
    r = measures(logs, map_dir, "scene-stop")
    m = r["measures"]
    assert m["n_stops"] == 1 and m["n_starts"] == 1
    assert 6.5 < m["longest_stop_s"] < 9
    assert m["near_traffic_light"] == 1.0
    assert m["n_peds_crossing"] >= 1


def test_agent_counts(logs, map_dir):
    m = measures(logs, map_dir, "scene-crowd")["measures"]
    assert m["n_peds"] >= 15
    assert m["n_peds_moving"] == 0
