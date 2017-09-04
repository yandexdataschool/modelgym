import numpy as np
from bson.son import SON
from skopt import gp_minimize

import modelgym
from modelgym import Trainer


class GPTrainer(Trainer):
    def __init__(self, n_estimators=5000, opt_evals=50, state=None, load_previous=False):
        super().__init__(n_estimators, opt_evals, state, load_previous)
        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.opt_evals, self.opt_eval_num = opt_evals, 0
        self.default_params, self.best_params = None, None
        self.best_n_estimators = None

    def fit_eval(self, model, dtrain, dtest, params=None, n_estimators=None, custom_metric=None):
        return super().fit_eval(model=model, dtrain=dtrain, dtest=dtest, params=params, n_estimators=n_estimators,
                                custom_metric=custom_metric)

    def crossval_optimize_params(self, model, cv_pairs, max_evals=None, verbose=True, batch_size=10,
                                 tracker=None):
        random_state = np.random.RandomState(1)
        max_evals = max_evals or self.opt_evals
        self.opt_eval_num, self.best_loss = 0, np.inf

        skoptParams = modelgym.util.hyperopt2skopt_space(model.space)

        _ = gp_minimize(
            func=lambda params: self.crossval_fit_eval(model=model, cv_pairs=cv_pairs, params=params, verbose=verbose),
            dimensions=(skoptParams.values()), random_state=random_state, n_calls=max_evals,
            n_jobs=max_evals - 1)

        best_hyper_params = {k: v for k, v in zip(skoptParams.keys(), _.x)}
        print(best_hyper_params)
        bst = 1 - _.fun
        print("Best accuracy score =", bst)
        ans = best_hyper_params.copy()
        ans['loss'] = bst
        return ans if not isinstance(ans, SON) else ans.to_dict()

    def crossval_fit_eval(self, model, cv_pairs, params=None, n_estimators=None, verbose=False):
        params = params or model.default_params
        n_estimators = n_estimators or self.n_estimators
        if (isinstance(params, list)):
            model.set_parameter(params)
            res = super().crossval_fit_eval(model=model, cv_pairs=cv_pairs, n_estimators=n_estimators,
                                            verbose=verbose)
        elif (isinstance(params, dict)):
            params = model.preprocess_params(params)
            res = super().crossval_fit_eval(model=model, cv_pairs=cv_pairs, params=params, n_estimators=n_estimators,
                                            verbose=verbose)
        else:
            raise ValueError()
        return res['loss']

    def print_result(self, result, name='', extra_keys=None):
        super().print_result(result=result, name=name, extra_keys=extra_keys)
