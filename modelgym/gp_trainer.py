import time
from functools import partial

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
        print(self)

    def crossval_optimize_params(self, model, X_train, y_train, cv_pairs, max_evals=None, verbose=True):
        max_evals = max_evals or self.gp_evals
        self.gp_eval_num, self.best_loss = 0, np.inf

        seed = 0
        np.random.seed(seed)

        params_fixed = {
            'objective': 'binary:logistic',
            'silent': 1,
            'seed': seed,
        }

        reg = modelgym.XGBModel(learning_task=TASK_CLASSIFICATION)
        reg.preprocess_params(params_fixed)

        # reg = XGBClassifier(**params_fixed)
        fn = partial(self.crossval_fit_eval, model, cv_pairs, verbose=verbose)

        def objective(params):
            """ Wrap a cross validated inverted `accuracy` as objective func """
            # reg.set_params(**{k: p for k, p in zip(space.keys(), params)})
            pr = {k: p for k, p in zip(space.keys(), params)}
            print(pr)
            reg.set_params(**pr)
            return 1 - np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy'))

        # res_gp = gp_minimize(objective, space.values(), n_calls=50, random_state=seed)
        print(model.space.values())
        _ = gp_minimize(func=lambda params: self.crossval_fit_eval(cv_pairs=cv_pairs, params=params, verbose=verbose,model=model),
                        dimensions=model.defspace.values(), random_state=0, n_calls=50)
        # _ = gp_minimize(func=fn, dimensions=model.defspace.values(), random_state=0, n_calls=50)
        best_hyper_params = {k: v for k, v in zip(model.defspace.keys(), _.x)}
        print(best_hyper_params)
        bst = 1 - _.fun
        print("Best accuracy score =", bst)
        return bst if not isinstance(bst, SON) else bst.to_dict()

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        params = params or model.default_params
        print(params)
        n_estimators = n_estimators or self.n_estimators
        # params = model.default_params
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
                     'params': params.copy()}
        self.best_loss = min(self.best_loss, cv_result['loss'])
        self.gp_eval_num += 1
        cv_result.update({'gp_eval_num': self.gp_eval_num, 'best_loss': self.best_loss})

        if verbose:
            print('[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                self.gp_eval_num, self.gp_evals, eval_time,
                model.metric, cv_result['loss'], self.best_loss))

        return cv_result['loss']

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
