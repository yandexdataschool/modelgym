from modelgym.trainers.trainer import Trainer, eval_metrics
from modelgym.utils.model_space import process_model_spaces
from modelgym.utils import cat_preprocess_cv

from functools import partial
from hyperopt import fmin, Trials, STATUS_OK, tpe
import numpy as np

class HyperoptTrainer(Trainer):
    def __init__(self, model_spaces, algo, tracker=None):
        if algo is None:
            raise ValueError("algo should not be None")
        self.model_spaces = process_model_spaces(model_spaces)
        self.tracker = tracker
        self.state = None
        self.algo = algo

    # TODO: consider different batch_size for different models
    def crossval_optimize_params(self, opt_metric, dataset, cv=3,
                                 opt_evals=50, metrics=None, batch_size=10,
                                 verbose=False,
                                 one_hot_max_size=10, cat_preprocess=True):
        if metrics is None:
            metrics = []

        if self.tracker is not None:
            self.state = self.tracker.load_state()

        if self.state is None:
            self.state = {name: Trials() for name in self.model_spaces}

        metrics.append(opt_metric)

        if isinstance(cv, int):
            cv = dataset.cv_split(cv)

        for name, state in self.state.items():
            model_space = self.model_spaces[name]

            print(model_space.model_class)

            learning_task =  model_space.model_class.get_learning_task()

            if cat_preprocess:
                cat_preprocess_cv(
                        cv, one_hot_max_size, learning_task)

            if len(state) == opt_evals:
                continue

            fn = lambda params: HyperoptTrainer.crossval_fit_eval(
                model_type=model_space.model_class,
                params=params,
                cv=cv, metrics=metrics, verbose=verbose
            )

            for i in range(0, opt_evals, batch_size):
                current_evals = min(batch_size, opt_evals - i)
                best = fmin(fn=fn,
                            space=model_space.space,
                            algo=self.algo,
                            max_evals=(i + current_evals),
                            trials=state)
                if self.tracker is not None:
                    self.tracker.save_state(self.state)

    def get_best_results(self):
        return {name: {"result" : trials.best_trial["result"],
                       "model_space" : self.model_spaces[name]}
                for (name, trials) in self.state.items()}

    @staticmethod
    def crossval_fit_eval(model_type, params, cv, metrics, verbose):
        metric_cv_results = []
        losses = []
        for dtrain, dtest in cv:
            eval_result = \
                eval_metrics(model_type, params, dtrain, dtest, metrics)
            metric_cv_results.append(eval_result)

            if metrics[-1].is_min_optimal:
                loss = eval_result[metrics[-1].name]
            else:
                loss = -eval_result[metrics[-1].name]
            losses.append(loss)

        return {
            "loss": np.mean(losses),
            "loss_variance": np.std(losses),
            "metric_cv_results": metric_cv_results,
            "params": params.copy(),
            "status": STATUS_OK
        }

class TpeTrainer(HyperoptTrainer):
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, algo=tpe.suggest, tracker=tracker)

class RandomTrainer(HyperoptTrainer):
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, algo=tpe.suggest, tracker=tracker)
