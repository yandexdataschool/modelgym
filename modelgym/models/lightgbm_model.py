import lightgbm as lgb
import numpy as np

from modelgym.models import Model, LearningTask
from hyperopt import hp
from hyperopt.pyll.base import scope


class LGBMClassifier(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict or None): parameters for model.
                             If None default params are fetched.
            learning_task (str): set type of task(classification, regression, ...)
        """

        if params is None:
            params = {}

        objective = 'binary'
        metric = 'binary_logloss'
        if params.get('num_class', 2) > 2:
            # change default objective
            objective = 'multiclass'
            metric = 'multi_logloss'

        self.params = {'objective': objective, 'metric': metric}
        self.params.update(params)
        self.n_estimators = self.params.pop('n_estimators', 1)
        self.model = None

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, dataset):
        return lgb.Dataset(data=dataset.X,
                           label=dataset.y,
                           categorical_feature=dataset.cat_cols)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        dtrain = self._convert_to_dataset(dataset)
        self.model = lgb.train(self.params, dtrain,
                               num_boost_round=self.n_estimators,
                               verbose_eval=False)
        return self

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.

        """
        assert self.model, "model is not fitted"
        return self.model.save_binary(filename)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        model = lgb.Booster(model_file=filename)
        new_model = LGBMClassifier()  # idk how to pass params yet
        new_model._set_model(model)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        prediction = self.model.predict(dataset.X)
        if self.params.get('num_class', 2) == 2:
            return np.round(prediction).astype(int)

        return np.argmax(prediction, axis=-1).astype(int)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        assert self.is_possible_predict_proba(), "Model cannot predict probability distribution"
        return self.model.predict(dataset.X)

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """
        return {
          'learning_rate':           hp.loguniform('learning_rate', -7, 0),
          'num_leaves':              scope.int(hp.qloguniform('num_leaves', 1, 7, 1)),
          'feature_fraction':        hp.uniform('feature_fraction', 0.5, 1),
          'bagging_fraction':        hp.uniform('bagging_fraction', 0.5, 1),
          'min_data_in_leaf':        scope.int(hp.qloguniform('min_data_in_leaf', 0, 6, 1)),
          'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
          'lambda_l1':               hp.loguniform('lambda_l1', -16, 2),
          'lambda_l2':               hp.loguniform('lambda_l2', -16, 2),
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION


class LGBMRegressor(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict or None): parameters for model.
                             If None default params are fetched.
            learning_task (str): set type of task(classification, regression, ...)
        """

        if params is None:
            params = {}

        self.params = {'objective': 'mean_squared_error', 'metric': 'l2'}
        self.params.update(params)
        self.n_estimators = self.params.pop('n_estimators', 1)
        self.model = None

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, dataset):
        return lgb.Dataset(data=dataset.X,
                           label=dataset.y,
                           categorical_feature=dataset.cat_cols)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        dtrain = self._convert_to_dataset(dataset)
        self.model = lgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
        return self

    def save_snapshot(self, filename):
        """
        Return:    serializable internal model state snapshot.

        """
        assert self.model, "model is not fitted"
        return self.model.save_binary(filename)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        model = lgb.Booster(model_file=filename)
        new_model = LGBMRegressor()  # idk how to pass params yet
        new_model._set_model(model)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        return self.model.predict(dataset.X)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
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
          'n_estimators':            scope.int(hp.quniform('n_estimators', 5, 10, 5)),
          'learning_rate':           hp.loguniform('learning_rate', -7, 0),
          'num_leaves':              scope.int(hp.qloguniform('num_leaves', 1, 7, 1)),
          'feature_fraction':        hp.uniform('feature_fraction', 0.5, 1),
          'bagging_fraction':        hp.uniform('bagging_fraction', 0.5, 1),
          'min_data_in_leaf':        scope.int(hp.qloguniform('min_data_in_leaf', 0, 6, 1)),
          'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
          'lambda_l1':               hp.loguniform('lambda_l1', -16, 2),
          'lambda_l2':               hp.loguniform('lambda_l2', -16, 2),
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.REGRESSION
