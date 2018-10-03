from modelgym.trainers.trainer import Trainer
from modelgym.utils.model_space import process_model_spaces
from modelgym.utils.evaluation import crossval_fit_eval
from skopt.optimizer import forest_minimize, gp_minimize, Optimizer
from modelgym.utils.util import log_progress
from modelgym.utils.dataset import  XYCDataset
from pandas import DataFrame
import tempfile
import os
import errno

import pickle
from pathlib import Path


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
        self.logs = []

    def crossval_optimize_params(self, opt_metric, dataset, cv=3,
                                 opt_evals=50, metrics=None,
                                 verbose=False, client=None, workers=1, timeout=100,
                                 **kwargs):
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
            self.ind2names[name] = [param.name for param in model_space.space]

        if metrics is None:
            metrics = [opt_metric]

        metrics.append(opt_metric)

        data_path = ""
        if client is None:
            cv = dataset.cv_split(cv)
        else:
            if isinstance(dataset, Path) or isinstance(dataset, str):
                if Path(dataset).expanduser().exists():
                    data_path = client.send_data(dataset)
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset)
            elif isinstance(dataset, DataFrame):
                with tempfile.NamedTemporaryFile() as temp:
                    dataset.to_csv(temp.name, sep=',')
                    data_path = client.send_data(temp.name)
            else:
                raise ValueError("Dataset has a wrong type {}".format(type(dataset)))

        for name, model_space in self.model_spaces.items():
            if client is None:
                fn = lambda params: self._eval_fn(
                    model_type=model_space.model_class,
                    params=params,
                    cv=cv, metrics=metrics, verbose=verbose, space_name=name
                )

                best = self.optimizer(fn, model_space.space,
                                  n_calls=opt_evals,
                                  n_random_starts=min(1, opt_evals))
            else:
                optimizer = Optimizer(
                    dimensions=model_space.space,
                    random_state=1,
                    acq_func="gp_hedge"
                )
                for _ in log_progress(range(opt_evals), every=1):
                    x = optimizer.ask(n_points=workers)  # x is a list of n_points points
                    x_named = []
                    for params in x:
                        x_named.append({self.ind2names[name][i]: params[i]
                                        for i in range(len(params))})
                    job_id_list = []
                    for model_params in x_named:
                        model_info = {"models": [{"type": model_space.model_class.__name__,
                                             "params": model_params}],
                                 "metrics": [m.name for m in metrics[1:]],
                                 "return_models": False,
                                 "cv": cv
                                 }
                        job_id_list.append(client.eval_model(model_info=model_info,
                                                        data_path=data_path))
                    result_list = client.gather_results(job_id_list, timeout=timeout)
                    if result_list == []:
                        continue
                    y_succeed = [result for result in result_list if not result is None]
                    x_succeed = [x_dot for i, x_dot in enumerate(x) if not result_list[i] is None]
                    self.logs += y_succeed
                    for res in y_succeed:
                        if self.best_results.get(name) is None:
                            self.best_results[name] = {"output": {"loss": 0}}
                        if res.get("output").get("loss") < self.best_results.get(name).get("output").get("loss"):
                            self.best_results[name] = res
                    if y_succeed != []:
                        best = optimizer.tell(x_succeed, [res.get("output").get("loss") for res in y_succeed])

        return self.best_results

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

    def get_best_model(self):
        loss = 0
        for (name, result) in self.best_results.items():
            if result.get("output").get("loss") < loss:
                loss = result.get("output").get("loss")
                best_m_path = result.get("result_model_path")
        with open(best_m_path, "rb") as f:
            best = pickle.load(f)
        return best

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
        if best["loss"] >= result["loss"]:
             self.best_results[space_name] = result
        self.logs.append(-result["loss"])
        return best["loss"]


class RFTrainer(SkoptTrainer):
    """RFTrainer is a SkoptTrainer, using Sequential optimisation using decision trees"""

    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, forest_minimize, tracker)


class GPTrainer(SkoptTrainer):
    """GPTrainer is a SkoptTrainer, using Bayesian optimization using Gaussian Processes."""

    def __init__(self, model_spaces, tracker=None):
        super().__init__(model_spaces, gp_minimize, tracker)
