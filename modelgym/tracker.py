import os
import pickle
from datetime import datetime

from hyperopt.mongoexp import MongoTrials
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class ProgressTracker(object):
    '''base class for storing model training state'''

    def __init__(self, model_name=None, config_key=None):
        self.model_name = model_name
        self.config_key = config_key
        self.schema = ('default_cv', 'default_test', 'tuned_cv', 'tuned_test', 'trials', '_model_name', '_config_key')
        self.state = dict([(k, None) for k in self.schema])
        self.state['_model_name'] = self.model_name
        self.state['_config_key'] = self.config_key

    def get_state(self, as_list=False):
        if as_list:
            return [self.state[k] for k in self.schema if not k.startswith('_')]
        else:
            return self.state

    def get_trials(self):
        return self.state['trials']

    def _exclude_keys(self, c, exc):
        return dict([(k, c[k]) for k in c.keys() if k not in exc])

    def _enhance_results(self, results, **kwargs):
        res = dict(results, **kwargs)
        res['timestamp'] = datetime.utcnow()
        return res

    def _update_state(self, kwargs):
        for k in self.schema:
            if k in kwargs:
                if isinstance(self.state[k], dict):
                    self.state[k] = self._exclude_keys(kwargs[k], ['bst'])
                else:
                    self.state[k] = kwargs[k]

    def save_state(self, **kwargs):
        raise NotImplementedError('Method save_state is not implemented.')

    def load_state(self, as_list=False):
        raise NotImplementedError('Method load_state is not implemented.')


class ProgressTrackerFile(ProgressTracker):
    ''' store model training state to file'''

    def __init__(self, results_dir, config_key=None, model_name=None):
        super(ProgressTrackerFile, self).__init__(model_name=model_name, config_key=config_key)
        self.results_dir = results_dir

    def _get_results_dir(self, results_dir=None):
        if results_dir is None:
            results_dir = self.results_dir
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        return results_dir

    def _get_tracker_file(self):
        results_dir = self._get_results_dir()
        return "%s/tracker_%s_%s.pickle" % (results_dir, self.config_key, self.model_name)

    def save_state(self, **kwargs):
        assert 'trials' not in kwargs or not isinstance(self.state['trials'],
                                                        MongoTrials), "this tracker cannot store mongoTrials"
        self._update_state(kwargs)
        path = self._get_tracker_file()

        with open(path, "wb") as fh:
            pickle.dump(self.state, fh)
            print("saved state to %s" % path)

    def load_state(self, as_list=False):
        path = self._get_tracker_file()
        if os.path.exists(path):
            with open(path, "rb") as fh:
                self.state = pickle.load(fh)
            print("loaded state from %s" % path)
        return self.get_state(as_list)


class ProgressTrackerMongo(ProgressTracker):
    def __init__(self, host, port, db, config_key=None, model_name=None):
        super(ProgressTrackerMongo, self).__init__(model_name=model_name, config_key=config_key)
        self.client = MongoClient(host, port)
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
        self._update_state(kwargs)
        state_without_trials = self._exclude_keys(self.state, 'trials')
        state_enhanced = self._enhance_results(state_without_trials,
                                               exp_key=self.model_name, config=self.config_key)

        self.mongo_collection.delete_many(dict(exp_key=self.model_name, config=self.config_key))
        self.mongo_collection.insert_one(state_enhanced)
        print("saved results to mongo %s" % self.mongo_collection.full_name)

    def load_state(self, as_list=False):
        r = self.mongo_collection.find_one({'config': self.config_key, 'exp_key': self.model_name})
        self.state.update(r)
        return self.get_state(as_list)
