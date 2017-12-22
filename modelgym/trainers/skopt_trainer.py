from modelgym.trainers.trainer import Trainer, eval_metrics
from modelgym.utils.model_space import process_model_spaces
from modelgym.utils import hyperopt2skopt_space
from skopt.optimizer import forest_minimize, gp_minimize
from modelgym.utils import cat_preprocess_cv

import numpy as np


class SkoptTrainer(Trainer):
    def __init__(self, model_spaces, optimizer, tracker=None):
        self.model_spaces = process_model_spaces(model_spaces)
        self.optimizer = optimizer
        self.best_results = {}
        self.ind2names = {}

    def crossval_optimize_params(self, opt_metric, dataset, cv=3,
                                 opt_evals=50, metrics=None, batch_size=10,
                                 verbose=False,
                                 one_hot_max_size=10, cat_preprocess=False):
        for name, model_space in self.model_spaces.items():
            skopt_space, ind2names = hyperopt2skopt_space(model_space.space)
            model_space.space = skopt_space
            self.ind2names[name] = ind2names

        if metrics is None:
            metrics = []

        metrics.append(opt_metric)

        if isinstance(cv, int):
            cv = dataset.cv_split(cv)

        if cat_preprocess not in [False, True]:
            if len(cat_preprocess) != len(self.model_spaces):
                raise ValueError('cat_preprocess must be True or False' +
                    'or bit-mask with len={}'.format(len(self.model_spaces)))
        else:
            cat_preprocess = np.zeros(len(self.model_spaces)) + cat_preprocess

        for model_index,[name, model_space] in \
                                enumerate(self.model_spaces.items()):

            model_space = self.model_spaces[name]

            learning_task = model_space.model_class.get_learning_task()

            if cat_preprocess[model_index]:
                cat_preprocess_cv(
                        cv, one_hot_max_size, learning_task)

            fn = lambda params: self.crossval_fit_eval(
                model_type=model_space.model_class,
                params=params,
                cv=cv, metrics=metrics, verbose=verbose, space_name=name
            )

            best = self.optimizer(fn, model_space.space,
                                  n_calls=opt_evals,
                                  n_random_starts=min(10, opt_evals))

            if best.fun > self.best_results[name]["loss"]:
                self.best_results[name] = Trainer.crossval_fit_eval(
                    model_space.model_class, best.x, cv, metrics, verbose)

    def get_best_results(self):
        return {name: {"result" : result,
                       "model_space" : self.model_spaces[name]}
                for (name, result) in self.best_results.items()}

    def crossval_fit_eval(self, model_type, params, cv, metrics, verbose,
                          space_name):
        params = {self.ind2names[space_name][i]: params[i]
                  for i in range(len(params))}
        result = Trainer.crossval_fit_eval(model_type, params, cv, metrics,
                                           verbose)
        loss = result["loss"]
        best = self.best_results.get(space_name, result)
        if best["loss"] <= result["loss"]:
            self.best_results[space_name] = result
        return best["loss"]


class RFTrainer(SkoptTrainer):
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, forest_minimize, tracker)


class GPTrainer(SkoptTrainer):
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, gp_minimize, tracker)
