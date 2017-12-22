from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
from six.moves import cPickle as pickle

from hyperopt.mongoexp import MongoTrials
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class Tracker(object):
    def __init__(self):
        self.state = {}

    def save_state(self):
        raise NotImplementedError('Method save_state is not implemented.')

    def load_state(self):
        raise NotImplementedError('Method load_state is not implemented.')


class LocalTracker(Tracker):
    def __init__(self, save_dir, suffix=None):
        super(LocalTracker, self).__init__()
        self._save_dir = save_dir
        self._save_file = os.path.join(
            save_dir,
            'tracker{}.pickle'.format('_' + suffix if suffix else '')
        )

    @staticmethod
    def check_exists(directory):
        if not os.path.exists(directory) or not os.path.isdir(directory):
            os.mkdir(directory)

    def save_state(self, state):
        self.state = state
        LocalTracker.check_exists(self._save_dir)
        with open(self._save_file, 'wb') as f:
            pickle.dump(self.state, f)

    def load_state(self): # encoding and other pickle params
        try:
            with open(self._save_file, 'rb') as f:
                self.state = pickle.load(f)
            return self.state
        except Exception as e:
            if isinstance(e, OSError):
                print ('no saved state found: {}'.format(self._save_file))
            else:
                print (e)
            self.state = None
            return self.state


class TrackerMongo(Tracker):  # dont really know what happens inside
    def __init__(self, host, port, db, config_key=None, model_name=None):
        super(TrackerMongo, self).__init__()
        self.client = MongoClient(host, port)
        self.model_name = model_name
        self.config_key = config_key

        try:
            self.client.admin.command('ismaster')
        except ConnectionFailure:
            print("Server not available")
            raise ConnectionFailure
        self.state['trials'] = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, db),
                                           exp_key=self.model_name)
        db = self.client[db]
        self.mongo_collection = db.results

    def save_state(self, **kwargs):
        state_without_trials = {k: v for k, v in self.state.items() if k != 'trials'}
        state_without_trials['exp_key'] = self.model_name
        state_without_trials['config'] = self.config_key
        state_without_trials['timestamp'] = datetime.utcnow()

        self.mongo_collection.delete_many(dict(exp_key=self.model_name, config=self.config_key))
        self.mongo_collection.insert_one(state_without_trials)
        print("saved results to mongo %s" % self.mongo_collection.full_name)

    def load_state(self, as_list=False):
        r = self.mongo_collection.find_one({'config': self.config_key, 'exp_key': self.model_name})
        self.state.update(r)
        return self.state
