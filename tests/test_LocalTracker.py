from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tempfile

import pytest

from modelgym.tracker import LocalTracker

TEST_PARAMS = ["results", "data"]


def test__get_tracker_file():
    for param in TEST_PARAMS:
        tracker = LocalTracker(param)
        expected = '{}/tracker{}.pickle'.format(param, '')
        ans = tracker._save_file
        assert expected == ans

    for param in TEST_PARAMS:
        tracker = LocalTracker(param, suffix='my')
        expected = '{}/tracker{}.pickle'.format(param, '_my')
        ans = tracker._save_file
        assert expected == ans


@pytest.mark.usefixtures("generate_trials")
def test_save_state(generate_trials):
    tmpdir = tempfile.mkdtemp()
    tracker = LocalTracker(tmpdir)
    # Ensure the file is read/write by the creator only
    saved_umask = os.umask(0o077)
    trials = generate_trials
    path = tracker._save_file
    try:
        tracker.state['trials'] = trials
        tracker.save_state()

        assert os.path.isfile(path)

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
    tracker = LocalTracker(tmpdir)
    # Ensure the file is read/write by the creator only
    saved_umask = os.umask(0o077)
    trials = generate_trials
    print (trials, type(trials))
    path = tracker._save_file

    def _exclude_keys(c, exc):
        return dict([(k, c[k]) for k in c.keys() if k not in exc])

    try:
        tracker.state['trials'] = trials
        tracker.save_state()

        assert os.path.isfile(path)

        loaded = tracker.load_state()
        assert 'trials' in loaded
        assert loaded == tracker.state

    except IOError as e:
        print('IOError {0}'.format(e))
    else:
        os.remove(path)
    finally:
        os.umask(saved_umask)
        os.rmdir(tmpdir)
