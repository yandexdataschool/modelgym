import numpy as np
from bson.son import SON
from skopt import gp_minimize, forest_minimize

import modelgym
from modelgym.trainers.base_trainer import BaseTrainer

class SkoptTrainer(BaseTrainer):
    def __init__(self, n_estimators=5000, opt_evals=50):
        super().__init__(n_estimators, opt_evals)
        self.optimizer = None

    def fit_eval(self, model, dtrain, dtest, params=None, n_estimators=None, custom_metric=None):
        return super().fit_eval(model=model, dtrain=dtrain, dtest=dtest, params=params, n_estimators=n_estimators,
                                custom_metric=custom_metric)

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None, verbose=True, batch_size=10,
                                 tracker=None):
        if self.optimizer is None:
            raise NotImplementedError()

        random_state = np.random.RandomState(1)
        max_evals = max_evals or self.evals
        self.eval_num, self.best_loss = 0, np.inf

        skoptParams = modelgym.util.hyperopt2skopt_space(model.space)
        _ = gp_minimize(
            func=lambda params: (print(params), self.crossval_fit_eval(model=model, cv_pairs=cv_pairs, params=params, verbose=verbose))[1],
            dimensions=(skoptParams.values()), random_state=random_state, n_calls=max_evals,
            n_jobs=max_evals - 1)

        best_hyper_params = {k: v for k, v in zip(skoptParams.keys(), _.x)}
        print('\tHYPER', best_hyper_params)
        bst = 1 - _.fun
        print("Best accuracy score =", bst)
        ans = {}
        ans['loss'] = bst
        ans['params'] = best_hyper_params
        return ans if not isinstance(ans, SON) else ans.to_dict()

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        return super().crossval_fit_eval(model, cv_pairs, params, n_estimators, verbose)

class GPTrainer(SkoptTrainer):
    def __init__(self, n_estimators=5000, opt_evals=50):
        super().__init__(n_estimators, opt_evals)
        self.optimizer = gp_minimize

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

class ForestTrainer(SkoptTrainer):
    def __init__(self, n_estimators=5000, opt_evals=50):
        super().__init__(n_estimators, opt_evals)
        self.optimizer = forest_minimize

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