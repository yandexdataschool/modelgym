import time
from functools import partial
from collections import defaultdict

import hyperopt
import numpy as np
from bson.son import SON
from hyperopt import fmin, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.mongoexp import MongoTrials
from modelgym.util import merge_two_dicts, calculate_custom_metrics, check_custom_metrics


class Trainer(object):
    def __init__(self, n_estimators=5000, opt_evals=50, state=None, load_previous=False):
        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.hyperopt_evals, self.hyperopt_eval_num = opt_evals, 0
        self.default_params, self.best_params = None, None
        self.best_n_estimators = None
    
    def fit_eval(self, model, dtrain, dtest, params=None, n_estimators=None, custom_metrics=None,
                 compute_additional_statistics=False):
        check_custom_metrics(custom_metrics)
        
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
   
        if custom_metrics is not None:
            for metric in custom_metrics:
                result[metric.name] = metric.calculate(model, bst, dtest, _dtest,
                                                       sample_weight=None)
            if compute_additional_statistics:
                # Compute custom metric statistics
                raw_values = defaultdict(list)
                n_splits = 5
                for dtest_splitted in dtest.split(n_splits):
                    _dtest_splitted = model.convert_to_dataset(dtest_splitted.X, 
                                                               dtest_splitted.y, dtest_splitted.cat_cols)
                    for metric in custom_metrics:
                        raw_values[metric.name].append(metric.calculate(model,
                                                                        bst,
                                                                        dtest_splitted,
                                                                       _dtest_splitted, 
                                                                        sample_weight=None))
                bounds = {}
                for metric in custom_metrics:
                    bounds[metric.name + '_bounds'] = result[metric.name] - np.min(raw_values[metric.name]),\
                                                      np.max(raw_values[metric.name]) - result[metric.name]
                result = merge_two_dicts(result, bounds)
                
        return result

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, 
                          verbose=False, custom_metrics=None):
        check_custom_metrics(custom_metrics)
        params = params or model.default_params
        n_estimators = n_estimators or self.n_estimators
        params = model.preprocess_params(params)
        evals_results, start_time = [], time.time()
        mean_evals_results = []
        std_evals_results = []
        custom_metrics_results = []
       
        for dtrain, dtest in cv_pairs:
            _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
            _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
            estimator, evals_result = model.fit(params, _dtrain, _dtest, n_estimators)
            if custom_metrics:
                custom_metrics_results.append(calculate_custom_metrics(custom_metrics, 
                                                                             model, estimator, dtest,
                                                                             _dtest))
            evals_results.append(evals_result)
            mean_evals_results.append(np.mean(evals_result))
            std_evals_results.append(np.std(evals_result))
        best_n_estimators = np.argmin(mean_evals_results)
        eval_time = time.time() - start_time
    
        cv_result = {'loss': mean_evals_results[best_n_estimators],
                     'loss_variance': std_evals_results[best_n_estimators],
                     'best_n_estimators': best_n_estimators + 1,
                     'eval_time': eval_time,
                     'status': STATUS_FAIL if np.isnan(mean_evals_results[best_n_estimators]) else STATUS_OK,
                     'params': params.copy()}
        if custom_metrics:
            for metric in custom_metrics:
                cv_result[metric.name] = custom_metrics_results[best_n_estimators][metric.name]
          
        self.best_loss = min(self.best_loss, cv_result['loss'])
        self.hyperopt_eval_num += 1
        if verbose:
            print('[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                self.hyperopt_eval_num, self.hyperopt_evals, eval_time,
                model.metric, cv_result['loss'], self.best_loss))

        return cv_result

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None, verbose=True, algo_name='tpe',
                                 batch_size=10, trials=None, tracker=None, custom_metrics=None):
        check_custom_metrics(custom_metrics)
        max_evals = max_evals or self.hyperopt_evals
        if trials is None:
            trials = Trials()
        algo = hyperopt.tpe.suggest
        if algo_name == 'random':
            algo = hyperopt.rand.suggest
        # algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=1)

        self.hyperopt_eval_num, self.best_loss = 0, np.inf
        random_state = np.random.RandomState(1)
        if isinstance(trials, MongoTrials):
            batch_size = max_evals  # no need for epochs
    
        for i in range(0, max_evals, batch_size):
            fn = partial(self.crossval_fit_eval, model, cv_pairs, verbose=verbose, custom_metrics=custom_metrics)
            # lambda params: self.run_cv(cv_pairs, dict(self.default_params, **params), verbose=verbose)
            n_jobs = min(batch_size, max_evals - i)
            best = fmin(fn=fn,
                        space=model.space,
                        algo=algo,
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

    def print_result(self, result, name='', extra_keys=None):
        print('\n%s:\n' % name)
        print('loss = %s' % (result['loss']))
        if 'best_n_estimators' in result.keys():
            print('best_n_estimators = %s' % result['best_n_estimators'])
        elif 'n_estimators' in result.keys():
            print('n_estimators = %s' % result['n_estimators'])
        if extra_keys is not None:
            for k in extra_keys:
                if k in result:
                    print("%s = %f" % (k, result[k]))
                    
        print('params = %s' % result['params'])
# class MongoTrainer(object):
