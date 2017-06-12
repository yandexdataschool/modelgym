import numpy as np
import os
import pickle
from hyperopt.mongoexp import MongoTrials

class ProgressTracker(object):
    ''' store model training state to file'''

    def __init__(self, model, config_key, results_dir, preload_cache=False):
        self.model = model
        self.results_dir = results_dir
        self.config_key = config_key # TODO; add to schema
        self.schema = ('default_cv', 'default_test', 'tuned_cv', 'tuned_test', 'trials')
        self.state = dict([(k, None) for k in self.schema])
        self.default, self.tuned, self.trials = None, None, None
        self.is_mongo = False # TODO: 


    def get_results_dir(self, results_dir=None):
        if results_dir is None: 
            results_dir = self.results_dir
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        return results_dir

        
    def get_tracker_file(self):
        exp_key = self.model.get_name()
        results_dir = self.get_results_dir()
        return None if self.is_mongo else "%s/tracker_%s_%s.pickle" % (results_dir, self.config_key, exp_key)


    # def save_results(default_cv_test, default_test, tuned_cv, tuned_test, trials):
    # def save_results(default_cv=None, default_test=None, tuned_cv=None, tuned_test=None, trials=None):
    def save_state(self, **kwargs):
        exp_key = self.model.get_name()

        assert 'trials' not in kwargs or not isinstance(self.state['trials'], MongoTrials), "this tracker cannot store mongoTrials"
        def exclude_keys(c, exc): return dict([(k, c[k]) for k in c.keys() if k not in exc])
        for k in self.schema:
            if k in kwargs:
                
                if isinstance(self.state[k], dict):
                    self.state[k] = exclude_keys(kwargs[k], ['bst'])
                else:
                    self.state[k] = kwargs[k]

        with open(self.get_tracker_file(), "wb") as fh:
            pickle.dump(self.state, fh)
            print("saved state to %s" % self.get_tracker_file())


    def get_state(self, as_list=False):
        if as_list:
            return [self.state[k] for k in self.schema]
        else:
            return self.state


    def load_state(self, as_list=False):
        if os.path.exists(self.get_tracker_file()):
            with open(self.get_tracker_file()) as fh:
                self.state = pickle.load(fh)
        return self.get_state(as_list)
        

class ProgressTrackerMongo(ProgressTracker):

    def save_state(self, default, tuned, mongo_collection=None):
        exp_key = model.get_name()
        def exclude_keys(c, exc): 
            return dict([(k, c[k]) for k in c.keys() if k not in exc])

        def enhance_results(results, **kwargs):
            res = dict(results, **kwargs)
            res['timestamp'] = datetime.datetime.utcnow()
            return res

        default_result = enhance_results(exclude_keys(default, ['bst']), exp_key=exp_key, config=CONFIG, flavour='default')
        tuned_result = enhance_results(exclude_keys(tuned, ['bst']), exp_key=exp_key, config=CONFIG, flavour='tuned')
        _ = mongo_collection.insert_one(default_result)
        _ = mongo_collection.insert_one(tuned_result)
        print "saved results to mongo %s" % mongo_collection.full_name
