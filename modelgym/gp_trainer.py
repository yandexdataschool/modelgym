import time

import numpy as np
from bson.son import SON
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize

import modelgym
from modelgym.model import TASK_CLASSIFICATION


class GPTrainer(object):
    def __init__(self, n_estimators=5000, gp_evals=50, state=None, load_previous=False):
        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.gp_evals, self.gp_eval_num = gp_evals, 0
        self.default_params, self.best_params = None, None
        self.best_n_estimators = None

    def print_result(self, result, name='', extra_keys=None):
        # TODO test
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

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None, verbose=True, batch_size=10,
                                 tracker=None):
        random_state = np.random.RandomState(1)
        max_evals = max_evals or self.gp_evals
        self.gp_eval_num, self.best_loss = 0, np.inf

        prgp = modelgym.util.process_params_gp(model)
        print("PRGP", prgp)

        _ = gp_minimize(
            func=lambda params: self.crossval_fit_eval(model=model, cv_pairs=cv_pairs, params=params, verbose=verbose),
            dimensions=(prgp.values()), random_state=random_state, n_calls=max_evals,
            n_jobs=max_evals - 1)

        best_hyper_params = {k: v for k, v in zip(prgp.keys(), _.x)}
        print(best_hyper_params)
        bst = 1 - _.fun
        print("Best accuracy score =", bst)
        ans = best_hyper_params.copy()
        ans['loss'] = bst
        return ans if not isinstance(ans, SON) else ans.to_dict()

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        params = params or model.default_params
        if (isinstance(params, list)):
            eta, max_depth, subsample, colsample_bytree, colsample_bylevel, min_child_weight, gamma, alpha, lambdax = \
                params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
            model.set_params(eta=eta,
                             max_depth=max_depth,
                             subsample=subsample,
                             colsample_bylevel=colsample_bylevel,
                             colsample_bytree=colsample_bytree,
                             min_child_weight=min_child_weight,
                             gamma=gamma,
                             alpha=alpha,
                             lambdax=lambdax
                             )
        n_estimators = n_estimators or self.n_estimators
        evals_results, start_time = [], time.time()
        if (isinstance(params, dict)):
            params = model.preprocess_params(params)

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
        loss = mean_evals_results[best_n_estimators - 1]

        self.best_loss = min(self.best_loss, loss)
        self.gp_eval_num += 1

        if verbose:
            print('[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                self.gp_eval_num, self.gp_evals, eval_time,
                model.metric, loss, self.best_loss))
        return loss

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
                  'eval_time': eval_time, 'params': params.copy()}

        if custom_metric is not None:
            if type(custom_metric) is not dict:
                raise TypeError("custom_metric argument should be dict")
            prediction = model.predict(bst, _dtest, dtest.X)  # TODO: why 2 args?
            for metric_name, metric_func in custom_metric.items():
                score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
                result[metric_name] = score

        return result
