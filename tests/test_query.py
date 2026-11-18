from scenario_search.query import normalize, numeric_score, parse


def tags(q):
    return parse(q).explain()["tags"]


def test_longest_phrase_wins():
    assert tags("many pedestrians at a crosswalk") == ["many_pedestrians", "crosswalk"]


def test_negation():
    assert tags("left turn without any cyclists") == ["turn_left", "NOT cyclist"]
    assert "NOT pedestrians" in tags("intersection with no pedestrians")


def test_numeric_filters_and_units():
    pq = parse("faster than 30 mph with at least 20 pedestrians")
    f = {x.measure: (x.op, round(x.value, 1)) for x in pq.numeric}
    assert f["max_speed_kmh"] == (">", 48.3)
    assert f["n_peds"] == (">=", 20.0)
    assert pq.terms == []


def test_residual_goes_to_open_vocab():
    pq = parse("bus stop next to a red building")
    assert pq.explain()["tags"] == ["bus_stop"]
    assert pq.residual == "next red building"


def test_numeric_score_is_monotone():
    assert numeric_score(60, ">", 50) > 0.99
    assert numeric_score(40, ">", 50) < 0.01
    assert numeric_score(40, "<", 50) > 0.99


def test_normalize_strips_punctuation():
    assert normalize("Left-turn, at NIGHT!") == "left-turn at night"
