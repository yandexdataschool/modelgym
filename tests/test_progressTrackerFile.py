import pytest
from modelgym.tracker import ProgressTrackerFile


def test__get_results_dir():
    test_params = ["results", "data"]
    for param in test_params:
        tracker = ProgressTrackerFile(param)
        ans = tracker._get_results_dir()
        assert param == ans


def test__get_tracker_file():
    test_params = ["results", "data"]
    for param in test_params:
        tracker = ProgressTrackerFile(param)
        expected = "%s/tracker_%s_%s.pickle" % (tracker._get_results_dir(), tracker.config_key, tracker.model_name)
        ans = tracker._get_tracker_file()
        assert expected == ans


def test_save_state():
    return 0


def test_load_state():
    return 0
