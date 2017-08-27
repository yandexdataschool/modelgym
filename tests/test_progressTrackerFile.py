import os.path
import pickle

import pytest

from modelgym.tracker import ProgressTrackerFile

TEST_PARAMS = ["results", "data"]


def test__get_results_dir():
    for param in TEST_PARAMS:
        tracker = ProgressTrackerFile(param)
        ans = tracker._get_results_dir()
        assert param == ans


def test__get_tracker_fil():
    for param in TEST_PARAMS:
        tracker = ProgressTrackerFile(param)
        expected = "%s/tracker_%s_%s.pickle" % (tracker._get_results_dir(), tracker.config_key, tracker.model_name)
        ans = tracker._get_tracker_file()
        assert expected == ans
        assert os.path.isfile(expected)


@pytest.mark.usefixtures("generate_trials")
def test_save_state(generate_trials):
    for param in TEST_PARAMS:
        tracker = ProgressTrackerFile(param)
        trials = generate_trials
        tracker.save_state(trials=trials)
        fname = tracker._get_tracker_file()
        assert os.path.isfile(fname)


def test_load_state():
    for param in TEST_PARAMS:
        tracker = ProgressTrackerFile(param)
        assert os.path.exists(tracker._get_tracker_file())
        with open(tracker._get_tracker_file(), "rb") as fh:
            tracker.state = pickle.load(fh)
            assert tracker.load_state(as_list=True) == tracker.get_state(as_list=True)
            assert tracker.load_state(as_list=False) != tracker.get_state(as_list=True)


@pytest.fixture()
def generate_trials():
    import math
    from hyperopt import fmin, tpe, hp, Trials
    trials = Trials()
    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
    yield trials
