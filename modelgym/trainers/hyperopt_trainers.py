import time
from functools import partial

import hyperopt
import numpy as np
from bson.son import SON
from hyperopt import fmin, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.mongoexp import MongoTrials

from modelgym.trainers.base_trainer import BaseTrainer


class HyperoptTrainer(BaseTrainer):
    def __init__(self, n_estimators=5000, opt_evals=50):
        super().__init__(n_estimators, opt_evals)
        self.algo = None

    def fit_eval(self, model, dtrain, dtest, params=None, n_estimators=None, custom_metric=None):
        return super().fit_eval(model=model, dtrain=dtrain, dtest=dtest, params=params, n_estimators=n_estimators,
                                custom_metric=custom_metric)

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None,verbose=True,
                                 batch_size=10, trials=None, tracker=None):
        if self.algo is None:
            raise NotImplementedError()
        max_evals = max_evals or self.evals
        if trials is None:
            trials = Trials()
        
        self.eval_num, self.best_loss = 0, np.inf
        random_state = np.random.RandomState(1)
        if isinstance(trials, MongoTrials):
            batch_size = max_evals  # no need for epochs

        for i in range(0, max_evals, batch_size):
            fn = partial(self.crossval_fit_eval, model, cv_pairs, verbose=verbose)
            # lambda params: self.run_cv(cv_pairs, dict(self.default_params, **params), verbose=verbose)
            n_jobs = min(batch_size, max_evals - i)
            best = fmin(fn=fn,
                        space=model.space,
                        algo=self.algo,
                        max_evals=(i + n_jobs),
                        trials=trials,
                        rstate=random_state)
            self.best_params = trials.best_trial['result']['params']
            if isinstance(self.best_params, SON):
                self.best_params = self.best_params.to_dict()
            self.best_n_estimators = trials.best_trial['result']['best_n_estimators']
            random_state = None
            if tracker is not None:
                tracker.save_state(trials=trials)

        bst = trials.best_trial['result']
        return bst if not isinstance(bst, SON) else bst.to_dict()

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        return super().crossval_fit_eval(model, cv_pairs, params, n_estimators, verbose)

    def print_result(self, result, name='', extra_keys=None):
        super().print_result(result, name='', extra_keys=None)

class TPETrainer(HyperoptTrainer):
    def __init__(self, n_estimators=5000, opt_evals=50):
        super().__init__(n_estimators, opt_evals)
        self.algo = hyperopt.tpe.suggest

    def fit_eval(self, model, dtrain, dtest, params=None, n_estimators=None, custom_metric=None):
        return super().fit_eval(model=model, dtrain=dtrain, dtest=dtest, params=params, n_estimators=n_estimators,
                                custom_metric=custom_metric)

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None, verbose=True, batch_size=10,
                                 tracker=None):
        return super().crossval_optimize_params(model, cv_pairs, max_evals=None, verbose=True, batch_size=10,
                                 tracker=None)

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        return super().crossval_fit_eval(model, cv_pairs, params, n_estimators, verbose)

    def print_result(self, result, name='', extra_keys=None):
        super().print_result(result, name='', extra_keys=None)

class RandomTrainer(HyperoptTrainer):
    def __init__(self, n_estimators=5000, opt_evals=50):
        super().__init__(n_estimators, opt_evals)
        self.algo = hyperopt.random.suggest

    def fit_eval(self, model, dtrain, dtest, params=None, n_estimators=None, custom_metric=None):
        return super().fit_eval(model=model, dtrain=dtrain, dtest=dtest, params=params, n_estimators=n_estimators,
                                custom_metric=custom_metric)

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None, verbose=True, batch_size=10,
                                 tracker=None):
        return super().crossval_optimize_params(model, cv_pairs, max_evals=None, verbose=True, batch_size=10,
                                 tracker=None)

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        return super().crossval_fit_eval(model, cv_pairs, params, n_estimators, verbose)

    def print_result(self, result, name='', extra_keys=None):
        super().print_result(result, name='', extra_keys=None)