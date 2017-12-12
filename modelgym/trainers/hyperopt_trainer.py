from modelgym.trainers.trainer import Trainer

from functools import partial
from hyperopt import fmin, Trials, STATUS_OK


class HyperoptTrainer(Trainer):
    def __init__(self, model_spaces, algo, tracker=None):
        self.model_spaces = model_spaces
        self.tracker = tracker
        self.state = None
        self.algo = algo

    # TODO: consider different batch_size for different models
    def crossval_optimize_params(self, opt_metric, dataset, cv=3, 
                                 opt_evals=None, metrics=[], batch_size=10,
                                 verbose=False):
        if self.tracker is not None:
            self.state = self.tracker.load_state()
        
        if self.state is None:
            self.state = [Trials() for _ in self.model_spaces]

        metrics.append(opt_metric)

        if isinstance(cv, int):
            cv = dataset.split(cv)

        for model_num, model_space in enumerate(self.model_spaces):
            if len(self.state[model_num].results) == opt_evals:
                continue

            fn = partial(HyperoptTrainer.crossval_fit_eval(
                model_type=model_space.model_type,
                cv=cv, metrics=metrics, verbose=verbose)
            )

            for i in range(0, opt_evals, batch_size):
                current_evals = min(batch_size, opt_evals - i)
                best = fmin(fn=fn,
                            space=self.model_space,
                            algo=self.algo,
                            max_evals=(i + current_evals),
                            trials=self.state[model_num])
                if self.tracker is not None:
                    tracker.save_state(self.state)

    def get_best_results():
        return [trials.best_trial["result"] for trial in self.state]

    @staticmethod
    def crossval_fit_eval(model_type, params, cv, metrics, verbose):
        metric_cv_results = []
        losses = []
        for dtrain, dtest in cv:
            eval_result = \
                Trainer.fit_eval(model_type, params, dtrain, dtest, metrics)
            metric_cv_results.append(eval_result)
            losses.append(eval_result[metrics[-1].name])
        return {
            "loss": np.mean(losses),
            "loss_variance": np.std(losses),
            "metric_cv_results": metric_cv_results,
            "params": params.copy(),
            "status": STATUS_OK
        }