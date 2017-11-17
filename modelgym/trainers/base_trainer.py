import time
from functools import partial

import hyperopt
import numpy as np
from bson.son import SON
from hyperopt import fmin, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.mongoexp import MongoTrials


class BaseTrainer(object):
    def __init__(self, n_estimators=5000, opt_evals=50):
        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.evals, self.eval_num = opt_evals, 0
        self.default_params, self.best_params = None, None
        self.best_n_estimators = None

    def fit_eval(self, model, dtrain, dtest, params=None, n_estimators=None, custom_metric=None):
        params = params or self.best_params or self.default_params
        n_estimators = n_estimators or self.best_n_estimators or self.n_estimators
        if params == None:
            params = model.default_params
        params = model.preprocess_params(params)
        start_time = time.time()
        _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
        _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
        bst, evals_result = model.fit(params, _dtrain, _dtest, n_estimators)
        eval_time = time.time() - start_time

        result = {'loss': evals_result[-1], 'bst': bst, 'n_estimators': n_estimators,
                  'eval_time': eval_time, 'status': STATUS_OK, 'params': params.copy()}

        if custom_metric is not None:
            if type(custom_metric) is not dict:
                raise TypeError("custom_metric argument should be dict")
            prediction = model.predict(bst, _dtest, dtest.X)  # TODO: why 2 args?
            for metric_name, metric_func in custom_metric.items():
                score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
                result[metric_name] = score

        return result

    def _crossval_fit_eval_imlp(self, model, cv_pairs, n_estimators, verbose, params=None):
        params = params or model.default_params
        n_estimators = n_estimators or self.n_estimators
        params = model.preprocess_params(params)
        evals_results, start_time = [], time.time()
        mean_evals_results = []
        std_evals_results = []
        for dtrain, dtest in cv_pairs:
            _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
            _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
            _, evals_result = model.fit(params, _dtrain, _dtest, n_estimators)
            evals_results.append(evals_result)
            mean_evals_results.append(np.mean(evals_result))
            std_evals_results.append(np.std(evals_result))
        best_n_estimators = np.argmin(mean_evals_results) + 1
        eval_time = time.time() - start_time

        cv_result = {'loss': mean_evals_results[best_n_estimators - 1],
                     'loss_variance': std_evals_results[best_n_estimators - 1],
                     'best_n_estimators': best_n_estimators,
                     'eval_time': eval_time,
                     'status': STATUS_FAIL if np.isnan(mean_evals_results[best_n_estimators - 1]) else STATUS_OK,
                     'params': params.copy()}
        self.best_loss = min(self.best_loss, cv_result['loss'])
        self.eval_num += 1
        cv_result.update({'eval_num': self.eval_num, 'best_loss': self.best_loss})

        if verbose:
            print('[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                self.eval_num, self.evals, eval_time,
                model.metric, cv_result['loss'], self.best_loss))

        return cv_result

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        params = params or model.default_params
        n_estimators = n_estimators or self.n_estimators
        if (isinstance(params, list)):
            model.set_parameters(params)
            res = self._crossval_fit_eval_imlp(model=model, cv_pairs=cv_pairs,
                                               n_estimators=n_estimators, verbose=verbose)
        elif (isinstance(params, dict)):
            params = model.preprocess_params(params)
            res = self._crossval_fit_eval_imlp(model=model, cv_pairs=cv_pairs, params=params,
                                               n_estimators=n_estimators, verbose=verbose)
        else:
            raise ValueError()
        return res
        

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None, verbose=True,
                                 batch_size=10, trials=None, tracker=None):
        raise NotImplementedError()

    def print_result(self, result, name='', extra_keys=None):
        print('%s:\n' % name)
        print('loss = %s' % (result['loss']))
        if 'best_n_estimators' in result.keys():
            print('best_n_estimators = %s' % result['best_n_estimators'])
        elif 'n_estimators' in result.keys():
            print('n_estimators = %s' % result['n_estimators'])
        print('params = %s' % result['params'])
        if extra_keys is not None:
            for k in extra_keys:
                if k in result:
                    print("%s = %f" % (k, result[k]))

