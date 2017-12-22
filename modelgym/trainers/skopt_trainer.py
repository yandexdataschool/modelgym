from modelgym.trainers.trainer import Trainer
from modelgym.utils.model_space import process_model_spaces
from modelgym.utils import hyperopt2skopt_space
from modelgym.utils.evaluation import crossval_fit_eval
from skopt.optimizer import forest_minimize, gp_minimize
from modelgym.utils import cat_preprocess_cv

import numpy as np


class SkoptTrainer(Trainer):
    """SkoptTrainer is a class for models hyperparameter optimization, based on skopt library"""

    def __init__(self, model_spaces, optimizer, tracker=None):
        """
        Args:
            model_spaces (list of modelgym.models.Model or modelgym.utils.ModelSpaces): list of model spaces
                (model classes and parameter spaces to look in). If some list item is Model, it is
                converted in ModelSpace with default space and name equal to model class __name__
            optimizer (function, e.g forest_minimize or gp_minimize
            tracker (modelgym.trackers.Tracker, optional): ignored
        Raises:
            ValueError if there are several model_spaces with similar names
        """
        super().__init__(model_spaces, tracker)
        self.model_spaces = process_model_spaces(model_spaces)
        self.optimizer = optimizer
        self.best_results = {}
        self.ind2names = {}

    def crossval_optimize_params(self, opt_metric, dataset, cv=3,
                                 opt_evals=50, metrics=None,
                                 verbose=False, **kwargs):
        """Find optimal hyperparameters for all models

        Args:
            opt_metric (modelgym.metrics.Metric): metric to optimize
            dataset (modelgym.utils.XYCDataset or None): dataset
            cv (int or list of tuples of (XYCDataset, XYCDataset)): if int, then number of cross-validation folds or
                cross-validation folds themselves otherwise.
            opt_evals (int): number of cross-validation evaluations
            metrics (list of modelgym.metrics.Metric, optional): additional metrics to evaluate
            verbose (bool): Enable verbose output.
            **kwargs: ignored
        Note:
            if cv is int, than dataset is split into cv parts for cross validation. Otherwise, cv folds are used.
        """

        for name, model_space in self.model_spaces.items():
            skopt_space, ind2names = hyperopt2skopt_space(model_space.space)
            model_space.space = skopt_space
            self.ind2names[name] = ind2names

        if metrics is None:
            metrics = []

        metrics.append(opt_metric)

        if isinstance(cv, int):
            cv = dataset.cv_split(cv)

        for name, model_space in self.model_spaces.items():

            fn = lambda params: self._eval_fn(
                model_type=model_space.model_class,
                params=params,
                cv=cv, metrics=metrics, verbose=verbose, space_name=name
            )

            best = self.optimizer(fn, model_space.space,
                                  n_calls=opt_evals,
                                  n_random_starts=min(10, opt_evals))

            if best.fun > self.best_results[name]["loss"]:
                self.best_results[name] = crossval_fit_eval(
                    model_space.model_class, best.x, cv, metrics, verbose)

    def get_best_results(self):
        """When training is complete, return best parameters (and additional information) for each model space

        Returns:
            dict of shape::

                {
                    name (str): {
                        "result": {
                            "loss": float,
                            "metric_cv_results": list,
                            "params": dict
                        },
                        "model_space": modelgym.utils.ModelSpace
                    }
                }

            name is a name of corresponding model_space,

            metric_cv_results contains dict's from metric names to calculated metric values for each fold in cv_fold,

            params is optimal parameters of corresponding model,

            model_space is corresponding model_space.
        """
        return {name: {"result": result,
                       "model_space": self.model_spaces[name]}
                for (name, result) in self.best_results.items()}

    def _eval_fn(self, model_type, params, cv, metrics, verbose, space_name):
        """Evaluates function to minimize and stores additional info (metrics, params) if it is current best result
        Args:
            model_type (type, subclass of Model)
            params (dict of str:obj): model parameters
            cv (list of tuple like (XYCDataset, XYCDataset)): cross validation folds
            metrics (list of modelgym.metrics.Metric): metrics to evaluate.
                Last metric is considered to be either loss (if metric.is_min_optimal is True) or -loss.
                Loss is the metric we want to minimize.
            verbose (bool): Enable verbose output.
            space_name (str): name of optimized model_space
        Returns:
            float: loss
        """
        params = {self.ind2names[space_name][i]: params[i]
                  for i in range(len(params))}
        result = crossval_fit_eval(model_type, params, cv, metrics, verbose)
        best = self.best_results.get(space_name, result)
        if best["loss"] <= result["loss"]:
            self.best_results[space_name] = result
        return best["loss"]


class RFTrainer(SkoptTrainer):
    """RFTrainer is a SkoptTrainer, using Sequential optimisation using decision trees"""
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, forest_minimize, tracker)


class GPTrainer(SkoptTrainer):
    """GPTrainer is a SkoptTrainer, using Bayesian optimization using Gaussian Processes."""
    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, gp_minimize, tracker)
