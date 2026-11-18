import numpy as np

from eval.ground_truth import gt_tags, split_of
from eval.run_eval import average_precision, ndcg_at, precision_at


def test_ap_perfect_and_worst():
    rel = np.array([1, 1, 0, 0], bool)
    assert average_precision(np.array([4, 3, 2, 1.0]), rel) == 1.0
    assert abs(average_precision(np.array([1, 2, 3, 4.0]), rel) - (1 / 3 + 2 / 4) / 2) < 1e-9


def test_ndcg_and_precision():
    rel = np.array([0, 1, 0, 1], bool)
    s = np.array([4, 3, 2, 1.0])
    assert precision_at(s, rel, 2) == 0.5
    assert 0 < ndcg_at(s, rel, 4) < 1


def test_gt_regexes():
    assert gt_tags("Night, rain, turn left, jaywalker") >= {"night", "rain", "turn_left", "jaywalker", "pedestrians"}
    assert "bus" not in gt_tags("peds at bus stop")
    assert "bus_stop" in gt_tags("peds at bus stop")
    assert "turn_right" in gt_tags("Right turn at intersection")


def test_split_is_deterministic():
    assert split_of("boston-seaport/2018-08-01") == split_of("boston-seaport/2018-08-01")
