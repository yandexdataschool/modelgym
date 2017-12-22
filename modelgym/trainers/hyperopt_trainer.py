from modelgym.trainers.trainer import Trainer, eval_metrics
from modelgym.utils.model_space import process_model_spaces
from modelgym.utils import cat_preprocess_cv

from hyperopt import fmin, Trials, STATUS_OK, tpe, rand
import numpy as np

class HyperoptTrainer(Trainer):
    def __init__(self, model_spaces, algo, tracker=None):
        self.model_spaces = process_model_spaces(model_spaces)
        self.tracker = tracker
        self.state = None
        self.algo = algo

    def crossval_optimize_params(self, opt_metric, dataset, cv=3,
                                 opt_evals=50, metrics=None, batch_size=10,
                                 verbose=False,
                                 one_hot_max_size=10, cat_preprocess=False):
        if metrics is None:
            metrics = []

        if self.tracker is not None:
            self.state = self.tracker.load_state()

        if self.state is None:
            self.state = {name: Trials() for name in self.model_spaces}

        metrics.append(opt_metric)

        if isinstance(cv, int):
            cv = dataset.cv_split(cv)

        if cat_preprocess not in [False, True]:
            if len(cat_preprocess) != len(self.state):
                raise ValueError('cat_preprocess must be True or False' +
                    'or bit-mask with len={}'.format(len(self.state)))
        else:
            cat_preprocess = np.zeros(len(self.state)) + cat_preprocess

        for model_index, [name, state] in enumerate(self.state.items()):
            model_space = self.model_spaces[name]

            learning_task = model_space.model_class.get_learning_task()

            if cat_preprocess[model_index]:
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
                fmin(fn=fn,
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
        result = Trainer.crossval_fit_eval(model_type, params, cv,
                                           metrics, verbose)
        result["status"] = STATUS_OK
        losses = [cv_result[metrics[-1].name]
                  for cv_result in result["metric_cv_results"]]
        result["loss_variance"] = np.std(losses)

        return result


class TpeTrainer(HyperoptTrainer):
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, algo=tpe.suggest, tracker=tracker)


class RandomTrainer(HyperoptTrainer):
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, algo=rand.suggest, tracker=tracker)
