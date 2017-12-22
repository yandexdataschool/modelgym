import xgboost as xgb
import numpy as np

from modelgym.models import Model, LearningTask
from hyperopt import hp
from hyperopt.pyll.base import scope


class XGBClassifier(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict): parameters for model.
        """

        if params is None:
            params = {}

        objective = 'binary:logistic'
        metric = 'logloss'
        if params.get('num_class', 2) > 2:
            # change default objective
            objective = 'multi:softprob'
            metric = 'mlogloss'

        self.params = {'objective': objective, 'eval_metric': metric,
                       'silent': 1}

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

    def _convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        dtrain = self._convert_to_dataset(dataset.X, dataset.y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
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
        booster = xgb.Booster()
        booster.load_model(filename)
        new_model = XGBClassifier()  # idk how to pass paarameters yet
        new_model._set_model(booster)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        xgb_dataset = xgb.DMatrix(dataset.X)
        if self.params['objective'] == 'multi:softprob':
            return self.model.predict(xgb_dataset).astype(int)
        prediction = np.round(self.model.predict(xgb_dataset)).astype(int)
        if self.params.get('num_class', 2) == 2:
            return prediction
        return np.argmax(prediction, axis=-1)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        if self.params['objective'] == 'multi:softprob':
            return False
        return True

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        xgb_dataset = xgb.DMatrix(dataset.X)
        assert self.is_possible_predict_proba(), "Model cannot predict probability distribution"
        return self.model.predict(xgb_dataset)

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """

        return {
            'eta':               hp.loguniform('eta', -7, 0),
            'max_depth':         scope.int(hp.quniform('max_depth', 2, 10, 1)),
            'n_estimators':      scope.int(hp.quniform('n_estimators', 100, 200, 100)),
            'subsample':         hp.uniform('subsample', 0.5, 1),
            'colsample_bytree':  hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight':  hp.loguniform('min_child_weight', -16, 5),
            'gamma':             hp.loguniform('gamma', -16, 2),
            'lambdax':           hp.loguniform('lambdax', -16, 2),
            'alpha':             hp.loguniform('alpha', -16, 2)
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION


class XGBRegressor(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict or None): parameters for model.
                             If None default params are fetched.
            learning_task (str): set type of task(classification, regression, ...)
        """

        if params is None:
            params = {}

        self.params = {'objective': 'reg:linear', 'eval_metric': 'rmse',
                       'silent': 1}
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

    def _convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        dtrain = self._convert_to_dataset(dataset.X, dataset.y, cat_cols=dataset.cat_cols)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
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
        booster = xgb.Booster()
        booster.load_model(filename)
        new_model = XGBRegressor()  # idk how to pass paarameters yet
        new_model._set_model(booster)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        xgb_dataset = xgb.DMatrix(dataset.X)
        return self.model.predict(xgb_dataset)

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
            'n_estimators':      scope.int(hp.quniform('n_estimators', 5, 10, 5)),
            'eta':               hp.loguniform('eta', -7, 0),
            'max_depth':         scope.int(hp.quniform('max_depth', 2, 10, 1)),
            'subsample':         hp.uniform('subsample', 0.5, 1),
            'colsample_bytree':  hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight':  hp.loguniform('min_child_weight', -16, 5),
            'gamma':             hp.loguniform('gamma', -16, 2),
            'lambdax':           hp.loguniform('lambdax', -16, 2),
            'alpha':             hp.loguniform('alpha', -16, 2)
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.REGRESSION
