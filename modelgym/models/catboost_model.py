import catboost as ctb
import numpy as np

from modelgym.models import Model, LearningTask
from hyperopt import hp
from hyperopt.pyll.base import scope


class CtBClassifier(Model):
    """
    Wrapper for `CatBoostClassifier
    <https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier-docpage/>`_
    """
    def __init__(self, params=None):
        """
        Args:
            params (dict): parameters for model.
        """

        if params is None:
            params = {}

        self.params = {
            'logging_level': 'Silent',
            'loss_function': 'Logloss'
        }

        classes_count = params.get('num_class', 2)
        if classes_count > 2:
            # change default objective
            self.params['loss_function'] = 'MultiClass'
            self.params['classes_count'] = classes_count

        self.params.update(params)
        self.model = None

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, data, label=None, cat_cols=None, **kwargs):
        """
        see https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_pool-docpage/
        Args:
            data: Dataset in the form of a two-dimensional feature matrix.
                or the path to the input file that contains the dataset description.
            label: The target values for the training dataset.
            cat_cols: A one-dimensional array of categorical columns indices.

        Return:
            CatBoost.Pool
        """
        return ctb.Pool(data, label=label, cat_features=cat_cols, **kwargs)

    def fit(self, dataset, weights=None, eval_dataset=None, **kwargs):
        """
        Args:
            dataset (XYCDataset): train
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
            eval_dataset: same as dataset
            kwargs: CatBoost.Pool kwargs if eval_dataset is None or
                ``{'train': train_kwargs, 'eval': eval_kwargs}`` otherwise

        Return:
            self
        """
        params = kwargs if kwargs else self.params.get('fit_kwargs', {})

        if eval_dataset is None:
            dtrain = self._convert_to_dataset(dataset.X, dataset.y, **params)
            self.model = ctb.CatBoostClassifier(**self.params).fit(dtrain)
        else:
            dtrain = self._convert_to_dataset(dataset.X, dataset.y, **params['train_kwargs'])
            deval = self._convert_to_dataset(eval_dataset.X, eval_dataset.y, **params['eval_kwargs'])
            self.model = ctb.CatBoostClassifier(**self.params).fit(dtrain, eval_set=deval)
        return self

    def save_snapshot(self, filename):
        """
        :return: serializable internal model state snapshot.

        """
        assert self.model, "model is not fitted"
        return self.model.save_model(filename)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        new_model = CtBClassifier()  # idk how to pass paarameters yet
        new_model._set_model(ctb.CatBoostClassifier().load_model(filename))
        return new_model

    def predict(self, dataset, **kwargs):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            kwargs: CatBoost.Pool kwargs
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        params = kwargs if kwargs else self.params.get('predict_kwargs', {})

        ctb_dataset = ctb.Pool(dataset.X, **params)
        return self.model.predict(ctb_dataset)

    def is_possible_predict_proba(self):
        """
        :return: bool, whether model can predict proba
        """
        return True

    def predict_proba(self, dataset, **kwargs):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            kwargs: CatBoost.Pool kwargs
        Return:
            np.array, shape (n_samples, n_classes)
        """
        params = kwargs if kwargs else self.params.get('predict_proba_kwargs', {})

        ctb_dataset = ctb.Pool(dataset.X, **params)
        assert self.is_possible_predict_proba(), "Model cannot predict probability distribution"
        proba = self.model.predict_proba(ctb_dataset)
        if self.params.get('num_class', 2) == 2:
            return proba[:, 1]
        return proba

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """
        return {
            'iterations': scope.int(hp.quniform('iterations', 100, 500, 100)),
            'depth': scope.int(hp.quniform('depth', 1, 11, 1)),
            'learning_rate': hp.loguniform('learning_rate', -5, -1),
            'rsm': hp.uniform('rsm', 0, 1),
            'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
            'l2_leaf_reg': scope.int(hp.quniform('l2_leaf_reg', 1, 10, 1)),
            'bagging_temperature': hp.uniform('bagging_temperature', 0, 2)
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION


class CtBRegressor(Model):
    """
    Wrapper for `CatBoostRegressor
    <https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier-docpage/>`_
    """
    def __init__(self, params=None):
        """
        Args:
            params (dict or None): parameters for model.
                             If None default params are fetched.
            learning_task (str): set type of task(classification, regression, ...)
        """

        if params is None:
            params = {}
        self.params = {
            'logging_level': 'Silent',
            'loss_function': 'RMSE'
        }

        self.params.update(params)
        self.model = None

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, data, label=None, cat_cols=None, **kwargs):
        """
        see https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_pool-docpage/
        Args:
            data: Dataset in the form of a two-dimensional feature matrix.
                or the path to the input file that contains the dataset description.
            label: The target values for the training dataset.
            cat_cols: A one-dimensional array of categorical columns indices.
        Return:
            CatBoost.Pool
        """
        return ctb.Pool(data, label=label, cat_features=cat_cols, **kwargs)

    def fit(self, dataset, weights=None, eval_dataset=None, **kwargs):
        """
        Args:
            dataset (XYCDataset) train
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
            eval_dataset: same as dataset
            kwargs: CatBoost.Pool kwargs if eval_dataset is None or
                ``{'train': train_kwargs, 'eval': eval_kwargs}`` otherwise
        Return:
            self
        """
        params = kwargs if kwargs else self.params.get('fit_kwargs', {})

        if eval_dataset is None:
            dtrain = self._convert_to_dataset(dataset.X, dataset.y, **params)
            self.model = ctb.CatBoostRegressor(**self.params).fit(dtrain)
        else:
            dtrain = self._convert_to_dataset(dataset.X, dataset.y, **params['train_kwargs'])
            deval = self._convert_to_dataset(eval_dataset.X, eval_dataset.y, **params['test_kwargs'])
            self.model = ctb.CatBoostRegressor(**self.params).fit(dtrain, eval_set=deval)
        return self

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.
        """
        assert self.model, "model is not fitted"
        return self.model.save_model(filename)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        new_model = CtBRegressor()  # idk how to pass paarameters yet
        new_model._set_model(ctb.CatBoostRegressor().load_model(filename))
        return new_model

    def predict(self, dataset, **kwargs):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            kwargs: CatBoost.Pool kwargs
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        params = kwargs if kwargs else self.params.get('predict_kwargs', {})

        ctb_dataset = ctb.Pool(dataset.X, **params)
        prediction = self.model.predict(ctb_dataset)
        return prediction

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False

    def predict_proba(self, dataset, **kwargs):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            kwargs: CatBoost.Pool kwargs
        Return:
            np.array, shape (n_samples, n_classes)
        """
        raise ValueError("Regressor can't predict proba")

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """
        return {
            'iterations': scope.int(hp.quniform('iterations', 5, 10, 5)),
            'depth': scope.int(hp.quniform('depth', 1, 11, 1)),
            'learning_rate': hp.loguniform('learning_rate', -5, -1),
            'rsm': hp.uniform('rsm', 0, 1),
            'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
            'l2_leaf_reg': scope.int(hp.quniform('l2_leaf_reg', 1, 10, 1)),
            'bagging_temperature': hp.uniform('bagging_temperature', 0, 2),
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.REGRESSION
