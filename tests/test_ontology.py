from collections import Counter

from scenario_search.ontology import TAG_BY_NAME, TAGS, score_tags


def test_tag_names_unique_and_phrases_not_shared():
    assert len(TAG_BY_NAME) == len(TAGS)
    counts = Counter(p for t in TAGS for p in t.phrases)
    assert [p for p, c in counts.items() if c > 1] == []


def test_scores_are_monotone_in_measure():
    t = TAG_BY_NAME["turn_left"]
    assert t.score({"turn_left_deg": 5}) < 0.01 < 0.99 < t.score({"turn_left_deg": 90})
    night = TAG_BY_NAME["night"]  # negative scale: lower sun => more night
    assert night.score({"sun_elevation_deg": -20}) > 0.99 and night.score({"sun_elevation_deg": 30}) < 0.01


def test_appearance_tags_have_no_rule_score():
    s = score_tags({})
    assert "rain" not in s and "turn_left" in s
