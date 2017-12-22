class Trainer(object):
    """Trainer is a base class for models hyperparameter optimization"""

    def __init__(self, model_spaces, tracker=None, **kwargs):
        """
        Args:
            model_spaces (list of modelgym.models.Model or modelgym.utils.ModelSpaces): list of model spaces
                (model classes and parameter spaces to look in). If some list item is Model, it is
                converted in ModelSpace with default space and name equal to model class __name__
            tracker (modelgym.trackers.Tracker, optional): tracker to save
                (and load, if there was any) optimization progress.
            **kwargs: additional Concrete-Trainer specific parameters
        Raises:
            ValueError if there are several model_spaces with similar names
        """
        pass

    def crossval_optimize_params(self, opt_metric, dataset, cv=3, opt_evals=50, metrics=None, verbose=False, **kwargs):
        """Find optimal hyperparameters for all models

        Args:
            opt_metric (modelgym.metrics.Metric): metric to optimize
            dataset (modelgym.utils.XYCDataset or None): dataset
            cv (int or list of tuples of (XYCDataset, XYCDataset)): if int, then number of cross-validation folds or
                cross-validation folds themselves otherwise.
            opt_evals (int): number of cross-validation evaluations
            metrics (list of modelgym.metrics.Metric, optional): additional metrics to evaluate
            verbose (bool): Enable verbose output.
            **kwargs: additional Concrete-Trainer specific parameters
        Note:
            if cv is int, than dataset is split into cv parts for cross validation. Otherwise, cv folds are used.
        """
        raise NotImplementedError()

    def get_best_results(self):
        """When training is complete, return best parameters (and additional information) for each model space
        Returns:
            dicts of shape: {
                name (str): {
                    "result": {
                        "loss": float,
                        "metric_cv_results": list,
                        "params": dict
                    },
                    "model_space": modelgym.utils.ModelSpace
                }
            }
            where name is a name of corresponding model_space,
            metric_cv_results contains dict's from metric names to calculated metric values for each fold in cv_fold
            params is optimal parameters of corresponding model
            model_space is corresponding model_space.

            Also, additional elements might appear in "result" dict, depending on Concrete Trainer
        """
        raise NotImplementedError()
