import os.path
import pickle
import tempfile

import pytest

from modelgym.tracker import ProgressTrackerFile

TEST_PARAMS = ["results", "data"]


def test__get_results_dir():
    for param in TEST_PARAMS:
        tracker = ProgressTrackerFile(param)
        ans = tracker._get_results_dir()
        assert param == ans


def test__get_tracker_file():
    for param in TEST_PARAMS:
        tracker = ProgressTrackerFile(param)
        expected = "%s/tracker_%s_%s.pickle" % (tracker._get_results_dir(), tracker.config_key, tracker.model_name)
        ans = tracker._get_tracker_file()
        assert expected == ans


@pytest.mark.usefixtures("generate_trials")
def test_save_state(generate_trials):
    tmpdir = tempfile.mkdtemp()
    tracker = ProgressTrackerFile(tmpdir)
    # Ensure the file is read/write by the creator only
    saved_umask = os.umask(0o077)
    trials = generate_trials
    path = tracker._get_tracker_file()
    try:
        with open(path, "w") as tmp:
            assert os.path.isfile(path)
            tracker.save_state(trials=trials)
    except IOError as e:
        print('IOError {0}'.format(e))
    else:
        os.remove(path)
    finally:
        os.umask(saved_umask)
        os.rmdir(tmpdir)


@pytest.mark.usefixtures("generate_trials")
def test_load_state(generate_trials):
    tmpdir = tempfile.mkdtemp()
    tracker = ProgressTrackerFile(tmpdir)
    # Ensure the file is read/write by the creator only
    saved_umask = os.umask(0o077)
    trials = generate_trials
    path = tracker._get_tracker_file()
    try:
        with open(path, "w"):
            assert os.path.isfile(path)
            tracker.save_state(trials=trials)
        with open(path, "rb") as tmp:
            tracker.state = pickle.load(tmp)
            assert tracker.load_state(as_list=True) == tracker.get_state(as_list=True)
            assert tracker.load_state(as_list=False) != tracker.get_state(as_list=True)
    except IOError as e:
        print('IOError {0}'.format(e))
    else:
        os.remove(path)
    finally:
        os.umask(saved_umask)
        os.rmdir(tmpdir)
